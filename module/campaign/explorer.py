"""统一镜头探索器：从当前镜头先追小地图提示，再由近及远螺旋移动，每屏同时找敌人、关闭机关并登记设施。

替代原来各管镜头和预算的三套程序：清怪巡检（ChapterSweep 蛇形）、设施巡检（FacilitySurvey 蛇形）、
找机关螺旋（SwitchSearch）。三套之间的衔接缝隙出过多次事故：
实录 2026-10-25（HARD 9）：小队旁的关闭机关已登记，但没有可点敌人时只做找敌人的巡检，机关一次没踩；
巡检找左边界时拖进雾里配准失败，连续三轮整轮中止，一行都没扫。

几何不确定只降级、不停止：配准测不到位移时按拖动增益估算继续，连续测不到（雾或地图外空白）就拖回
一格当边界；小队位置不明时按最后看到的位置推算避让。镜头移动只用于观察，点击仍要当前新鲜检测。
want 决定优先级：第一类目标一出现就返回；其余类型记下首次出现的航点，全部看完仍没有第一类时倒回确认。
"""

import time

import cv2
import numpy as np

from module.campaign.facility_memory import FacilityUnconfirmed, facility_frame, voted_facilities
from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_region, pan_scene
from module.campaign.scene_memory import stable_candidates
from module.campaign.vision.objects import detect_objects
from module.campaign.vision.perception import MINIMAP_TO_GROUND, minimap_squad_ground, squad_arrow
from module.campaign.vision.story_marker import story_markers
from module.logger import logger

# 单次拖动手势上限（手指方向，镜头反向移动）。实测 2026-10-22（22-27）：700px横拖画面只动
# 290～335px，520px竖拖只动120～280px且带横向偏移，因此按实测位移走到各航点，不按拖动次数。
HORIZONTAL_DRAG = 1100.
VERTICAL_DRAG = 600.
# 航点间距（客户区像素，镜头位移）。斜俯视镜头下画面上部常被前方建筑挡住，只有中下部约0.55屏
# 算可靠覆盖，竖向按它取间距；横向约0.8屏留两成重叠。
# 实录 2026-10-22（22-27）：开关落在画面上部被箱子挡住时读不到，落在中下部时置信度0.95。
SPIRAL_UNIT = np.array([1420., 550.])
# 螺旋圈数：5 圈 30 个航点，覆盖起点周围横向约 ±4 屏、竖向约 ±2.5 屏。
SPIRAL_RINGS = 5
# 目录线索（已知关闭机关的投影）要移到的画面位置，以及线索离画面多远内才去看。
HINT_VIEW = np.array([888., 640.])
HINT_MARGIN = 300.
HINT_PANS = 3
# 离航点不足约四分之一屏即算到达；每个航点最多拖几次；位移不足多少像素算撞到边界。
WAYPOINT_TOLERANCE = np.array([450., 250.])
WAYPOINT_PANS = 6
STUCK_SHIFT = 30.
# 同一航点连续几次测不到位移就当作雾边：拖回一格，不再往这个方向走。
FOG_MISSES = 2
# 主平移至少要这么多对双向匹配支持。
SHIFT_SUPPORT = 30
# 拖动增益初值（画面位移/手指位移），之后按每次实测更新。
INITIAL_GAIN = .45
# 每屏设施取帧数与投票门槛：先取两帧，两帧一致即结束；不一致才补到四帧投票。烟雾会让单帧漏检。
FACILITY_QUICK_FRAMES = 2
FACILITY_FRAMES = 4
FACILITY_VOTES = 2
FACILITY_KINDS = ('ground_mechanism', 'elevator', 'barrier')
QUEST_MARKER_KINDS = ('scene', 'cue')
# 小队推算位置离画面边缘超过这么远就不再逐次识别箭头（识别一次约1.3秒）。
SQUAD_MARGIN = 300.
# 小地图敌人提示换算到画面后，离最近脚圈超过这么远才算「画面里没看到」；每次探索最多追几次。
MINIMAP_RING_RADIUS = 150.
MINIMAP_CHASES = 3
MINIMAP_VIEW = np.array([888., 600.])


class SweepUnconfirmed(RuntimeError):
    """探索缺乏运动或边界证据；禁止把未检查区域当成没有敌人。"""
    reason = 'chapter_clearance_unconfirmed'


def spiral_cells(rings=4):
    """返回以起点为中心逐圈外扩的航点格（单位：SPIRAL_UNIT）：先南、东，再北、西，每两段边长加一。"""
    directions = ((0, 1), (1, 0), (0, -1), (-1, 0))
    cells, cell, length = [], np.zeros(2), 1
    for leg in range(rings*2):
        for _ in range(length):
            cell = cell + directions[leg % 4]
            cells.append(cell.copy())
        if leg % 2:
            length += 1
    return cells


def content_shift(before, after, drag, first=None, second=None):
    """两帧静态背景特征的主平移（客户区像素，画面内容的位移）；测不到返回None。

    实测 2026-10-22（22-27）：高架与高楼的视差让单应内点率只有17～28%，严格配准判失败，
    但双向匹配的平移大多顺着手指方向。取顺手指方向（余弦≥0.8）且增益0.1～2的匹配取中位数；
    大多数匹配几乎不动则判镜头未动（地图边界）。只用于决定下一次拖动，不用于点击或设施身份。
    """
    first = facility_frame(before, []) if first is None else first
    second = facility_frame(after, []) if second is None else second
    if (first.descriptors is None or second.descriptors is None
            or min(len(first.keys), len(second.keys)) < SHIFT_SUPPORT):
        return None
    matcher = cv2.BFMatcher()
    reverse = matcher.match(second.descriptors, first.descriptors)
    pairs = [pair[0] for pair in matcher.knnMatch(first.descriptors, second.descriptors, k=2)
             if len(pair) == 2 and pair[0].distance < .7*pair[1].distance
             and reverse[pair[0].trainIdx].trainIdx == pair[0].queryIdx]
    if len(pairs) < SHIFT_SUPPORT:
        return None
    shifts = (np.float32([second.keys[item.trainIdx] for item in pairs])
              - np.float32([first.keys[item.queryIdx] for item in pairs]))
    still = np.linalg.norm(shifts, axis=1) < STUCK_SHIFT
    if still.sum() >= max(SHIFT_SUPPORT, .5*len(shifts)):
        return np.zeros(2)
    drag = np.asarray(drag, float)
    gain = shifts @ drag / max(drag @ drag, 1.)
    cosine = shifts @ drag / np.maximum(np.linalg.norm(shifts, axis=1)*np.linalg.norm(drag), 1.)
    along = (cosine >= .8) & (gain >= .1) & (gain <= 2.)
    if along.sum() < SHIFT_SUPPORT:
        return None
    return np.median(shifts[along], axis=0)


def off_switches(frames):
    """多帧投票同位置、完整入屏、无同类竞争的关闭地面机关；开启机关和平台不在此列。"""
    return [item for item in voted_facilities(frames, FACILITY_VOTES)
            if item['kind'] == 'ground_mechanism' and item['state'] == 'off']


def frames_agree(frames):
    """两帧的完整入屏设施逐一对上（同类同状态≤12px），无漏检或闪现，不必再补帧投票。"""
    single = [len(voted_facilities([frame], 1)) for frame in frames]
    return len(set(single)) == 1 and len(voted_facilities(frames, len(frames))) == single[0]


def minimap_hints(field):
    """小地图上的敌人标记换算成画面地面坐标；小地图不可见或读不到返回空列表。

    小地图以镜头注视点为中心、比例固定（perception.MINIMAP_TO_GROUND，实测残差多数<15px）。
    只用来决定往哪看，不作为点击落点。
    实录 2026-10-25（HARD 9）：巡检多屏「场景目标 0、小地图提示 3」，原先只用来阻止报全清。
    """
    from module.campaign.vision.detection import detect_minimap
    from module.campaign.vision.ex_ring import minimap_enemy_hints
    from module.campaign.vision.minimap import map_visible
    panel = cv2.resize(field, (1920, 1080))[96:307, 25:243]
    if not map_visible(panel):
        return []
    points = [np.asarray(item.center, float) for item in detect_minimap(panel, ('minimap_enemy_normal',))]
    for item in minimap_enemy_hints(panel):
        point = np.asarray(item['position'], float)
        if not any(np.linalg.norm(point-known) < 6. for known in points):
            points.append(point)
    return [MINIMAP_TO_GROUND @ np.r_[point, 1.] for point in points]


class Explorer:
    """有界镜头探索；任何真实备战弹窗立即交回战斗流程，找到目标不等于开门或开战。

    返回 dict(state=entry/enemy/switch/exhausted, field=..., ...)：enemy 带 targets（新鲜双帧确认的
    可选脚圈），switch 带 target（两帧以上投票的关闭机关）。exhausted 附带覆盖统计，不声称本章已清空。
    """

    def __init__(self, runner, want=('enemy', 'switch'), catalog=None, rings=SPIRAL_RINGS, seconds=300.,
                 skip=(), goal=None):
        """want：按优先级排列的目标类型；catalog：设施目录（可为 None，只看新鲜检测）；
        skip：本受阻目标已试过且未解决的机关搜索键；goal：受阻目标键，用于目录里的已试记录。"""
        self.runner, self.session, self.catalog = runner, runner.session, catalog
        self.want, self.skip = tuple(want), set(skip)
        self.goal = goal if goal is not None else 'quest:'+str(getattr(runner, 'quest', None))
        self.waypoints = [np.zeros(2)]+[cell*SPIRAL_UNIT for cell in spiral_cells(rings)]
        self.seconds = seconds
        self.views, self.pans, self.catalog_errors, self.unmeasured, self.edges = 0, 0, 0, 0, 0
        # 镜头相对起点的累计位移（按实测累加，测不到时按增益估算）及上次取帧位置。
        self.camera, self.gain, self.looked = np.zeros(2), INITIAL_GAIN, None
        self.hinted, self.chases = set(), 0
        # 小队箭头上次所见的画面位置及当时的镜头位置；小队在探索期间不动，离屏后按镜头位移推算。
        self.squad, self.squad_camera = None, None
        # 上一次拖动后帧的配准特征，作为下一次拖动的前帧特征复用。
        self.features = None
        self.started = time.monotonic()

    # ----- 观察 -----

    def enemies(self, field):
        """本屏可选敌人：单帧没有脚圈就不再取第二帧；有脚圈时两帧唯一对应才算候选。"""
        runner = self.runner
        first = runner.memory.observe(field)
        if not any(runner.selectable(item) for item in first):
            return field, []
        self.session.pause(.25)
        fresh = runner.map_frame(track=False)
        if goto.battle_popup_score(fresh) > .8:
            return fresh, None
        second = runner.memory.observe(fresh)
        candidates = [item for item in stable_candidates(first, second) if runner.selectable(item)]
        return fresh, runner.drop_rejected(fresh, candidates)

    def facilities(self, field):
        """本屏设施多帧检测并登记目录；返回 (新鲜帧, 各帧检测, 已登记设施)，弹窗时帧检测为 None。"""
        frames = [detect_objects(field, FACILITY_KINDS)]
        fresh = field
        # 设施至少两帧：烟雾会让单帧漏检（实录21-32三帧漏一帧），不做敌人那样的单帧捷径。
        while len(frames) < FACILITY_FRAMES:
            if len(frames) >= FACILITY_QUICK_FRAMES and frames_agree(frames):
                break
            self.session.pause(.25)
            fresh = self.runner.map_frame(track=False)
            if goto.battle_popup_score(fresh) > .8:
                return fresh, None, []
            frames.append(detect_objects(fresh, FACILITY_KINDS))
        registered = []
        if self.catalog is not None and any(frames):
            try:
                registered = self.catalog.observe(fresh, voted_facilities(frames, FACILITY_VOTES))
            except FacilityUnconfirmed:
                self.catalog_errors += 1
        return fresh, frames, registered

    def switches(self, frames, registered):
        """本屏可用的关闭机关：对本受阻目标没试过、不在已跳过的搜索键里。"""
        result = []
        for item in off_switches(frames):
            key = ('switch', *np.round(np.asarray(item['position'])/80.).astype(int).tolist(),
                   *np.round(self.camera/80.).astype(int).tolist())
            known = [entry['facility_id'] for entry in registered if entry['kind'] == item['kind']
                     and np.linalg.norm(np.asarray(entry['position'])-item['position']) < 1.]
            if key in self.skip or (known and not self.catalog.eligible(known[0], self.goal)):
                continue
            result.append(dict(item, search_key=key, **({'facility_id': known[0]} if known else {})))
        return result

    def catalog_hints(self, field):
        """目录里已知、对本目标没试过的关闭机关，投影在画面附近（含画面外 HINT_MARGIN 内）。"""
        if self.catalog is None:
            return []
        low, high = -HINT_MARGIN, np.array([field.shape[1], field.shape[0]])+HINT_MARGIN
        return [dict(hint, confidence=0.) for hint in getattr(self.catalog, 'expected', [])
                if hint['facility_id'] not in self.hinted
                and np.all(np.asarray(hint['position']) >= low) and np.all(np.asarray(hint['position']) <= high)
                and self.catalog.eligible(hint['facility_id'], self.goal)]

    def look(self, step, goal=False):
        """在当前镜头看一屏；返回 dict(state, field, enemies, switches, hints, minimap)。

        state：entry（备战弹窗）／seen（有结果）／none。goal 为真表示镜头已在受阻目标旁，
        此处敌人多与目标同在障碍后，不收集敌人，机关按离任务菱形最近排序。
        """
        started = time.monotonic()
        self.looked = self.camera.copy()
        field = self.runner.map_frame(track=False)
        if goto.battle_popup_score(field) > .8:
            return dict(state='entry', field=field)
        enemies = []
        if 'enemy' in self.want and not goal:
            field, enemies = self.enemies(field)
            if enemies is None:
                return dict(state='entry', field=field)
        fresh, frames, registered = self.facilities(field)
        if frames is None:
            return dict(state='entry', field=fresh)
        if registered:
            from module.campaign.navigation.facility_map import record_map_positions
            # 新登记的设施补底图坐标；开合地图若让画面变了，本屏重看一次，不用旧帧像素点击。
            moved = record_map_positions(self.runner, self.catalog, registered, fresh)
            if moved is not None and not moved['unchanged']:
                return self.look(step, goal)
        switches = self.switches(frames, registered) if 'switch' in self.want else []
        hints = self.catalog_hints(fresh) if 'switch' in self.want and not switches and not goal else []
        minimap = [] if goal or 'enemy' not in self.want else self.minimap_targets(fresh, enemies)
        self.views += 1
        if goal:
            marks = [item['position'] for item in story_markers(fresh) if item['kind'] in QUEST_MARKER_KINDS]
            anchor = np.asarray(marks[0] if len(marks) == 1 else goto.VIEWPORT_PIXEL, float)
        else:
            arrow = squad_arrow(fresh)
            anchor = np.asarray(goto.VIEWPORT_PIXEL if arrow is None else arrow, float)
        switches.sort(key=lambda item: np.linalg.norm(np.asarray(item['position'])-anchor))
        self.session.record_evidence('explore_view', fresh, step=step, detections=frames[-1],
                                     enemies=[dict(id=item['id'], click=item['click']) for item in enemies],
                                     switches=switches, hints=hints, minimap=[point.tolist() for point in minimap],
                                     frames=len(frames), camera=np.round(self.camera).tolist(),
                                     elapsed=round(time.monotonic()-started, 2), arrays=dict(field=fresh))
        self.session.preview(fresh)
        logger.info(f'探索第 {step} 屏：可选敌人 {len(enemies)} 个，关闭机关 {len(switches)} 个，'
                    f'目录线索 {len(hints)} 个，画面外小地图提示 {len(minimap)} 个，'
                    f'镜头离起点 {np.round(self.camera).astype(int).tolist()}。')
        seen = bool(enemies or switches or hints or minimap)
        return dict(state='seen' if seen else 'none', field=fresh, enemies=enemies, switches=switches,
                    hints=hints, minimap=minimap)

    def minimap_targets(self, field, enemies):
        """小地图敌人提示里，画面上附近没有可选脚圈的（被挡住、在画面边缘或画面外）。"""
        try:
            points = minimap_hints(field)
        except (cv2.error, ValueError):
            return []
        clicks = [np.asarray(item['click'], float) for item in enemies]
        return [point for point in points
                if not any(np.linalg.norm(point-click) <= MINIMAP_RING_RADIUS for click in clicks)
                and np.linalg.norm(point-MINIMAP_VIEW) > WAYPOINT_TOLERANCE[1]]

    # ----- 镜头 -----

    def squad_ground(self, before):
        """拖动按下点要避开的小队地面坐标：箭头 → 小地图圆环 → 按镜头位移推算；推算已离屏返回None。"""
        if self.squad is not None:
            predicted = self.squad-(self.camera-self.squad_camera)
            size = np.array([before.shape[1], before.shape[0]], float)
            if np.any(predicted < -SQUAD_MARGIN) or np.any(predicted > size+SQUAD_MARGIN):
                return None
        arrow = squad_arrow(before)
        if arrow is not None:
            ground = np.asarray(arrow, float)+[0., goto.ARROW_GROUND_OFFSET]
        else:
            ground = minimap_squad_ground(before)
            if ground is None:
                # 实录 2026-10-22（22-36）：小队在穹顶里箭头被挡、也没有离屏头像；小队探索期间不动，按位移推算。
                return None if self.squad is None else self.squad-(self.camera-self.squad_camera)
        self.squad, self.squad_camera = np.asarray(ground, float), self.camera.copy()
        return self.squad

    def pan(self, drag, before):
        """发送一次观察用镜头手势并实测镜头位移；返回 (拖后帧, 镜头位移, 是否实测)。"""
        ground = self.squad_ground(before)
        self.session.begin_target('explore_pan')
        pan_scene(self.session, drag, squad_ground=ground, settle=None, paced=True)
        self.pans += 1
        after = self.runner.map_frame(track=False)
        first = self.features[1] if self.features is not None and self.features[0] is before else None
        second = facility_frame(after, [])
        self.features = (after, second)
        shift = content_shift(before, after, drag, first, second)
        if shift is None:
            self.unmeasured += 1
            self.session.record_evidence('explore_unmeasured', after, drag=np.round(drag).tolist(),
                                         arrays=dict(field=after, previous_field=before))
            return after, -drag*self.gain, False
        gain = float(shift @ drag / max(drag @ drag, 1.))
        if np.linalg.norm(shift) >= STUCK_SHIFT and .1 <= gain <= 2.:
            self.gain = .5*self.gain+.5*gain
        return after, -shift, True

    def drag_toward(self, error):
        """把镜头移动 error（客户区像素）所需的手指拖动，按单次上限和拖动区域缩放。"""
        drag = -np.asarray(error, float)/self.gain
        drag *= min(1., HORIZONTAL_DRAG/max(abs(drag[0]), 1.), VERTICAL_DRAG/max(abs(drag[1]), 1.))
        return drag*pan_region(drag)[1]

    def move_by(self, error, pans=None, tolerance=None):
        """按实测位移把镜头移动 error；撞边界（两次几乎不动）或进雾（连续测不到）就停在原处。"""
        pans = WAYPOINT_PANS if pans is None else pans
        tolerance = WAYPOINT_TOLERANCE if tolerance is None else tolerance
        target = self.camera+np.asarray(error, float)
        field, stuck, missed = self.runner.map_frame(track=False), 0, 0
        for _ in range(pans):
            remaining = target-self.camera
            if np.all(np.abs(remaining) <= tolerance):
                return True
            self.session.check()
            drag = self.drag_toward(remaining)
            field, moved, measured = self.pan(drag, field)
            if not measured:
                missed += 1
                if missed >= FOG_MISSES:
                    # 实录 2026-10-25（HARD 9）：拖进悬崖外的雾里配准失败；拖回一格当边界，不整轮中止。
                    self.edges += 1
                    logger.info('镜头连续测不到位移（可能拖进雾里或地图外），拖回一格并把这个方向当作边界。')
                    field, back, _ = self.pan(-drag, field)
                    self.camera = self.camera+moved+back
                    return False
                self.camera = self.camera+moved
                continue
            missed = 0
            self.camera = self.camera+moved
            stuck = stuck+1 if np.linalg.norm(moved) < STUCK_SHIFT else 0
            if stuck >= 2:
                self.edges += 1
                return False
        return False

    def fresh_view(self):
        """当前镜头离上次取帧处已超过约四分之一屏（或从未取帧），值得再看一屏。"""
        return self.looked is None or np.any(np.abs(self.camera-self.looked) > WAYPOINT_TOLERANCE)

    # ----- 流程 -----

    def pick(self, seen, kinds):
        """按 kinds 优先级从一屏结果里挑可返回的目标。"""
        for kind in kinds:
            if kind == 'enemy' and seen.get('enemies'):
                return dict(state='enemy', targets=seen['enemies'], field=seen['field'])
            if kind == 'switch' and seen.get('switches'):
                return dict(state='switch', target=seen['switches'][0], field=seen['field'])
        return None

    def chase(self, seen, step):
        """目录线索与小地图提示：把提示位置移到画面中下部再看一屏；返回新一屏结果或 None。"""
        if seen.get('hints'):
            hint = min(seen['hints'], key=lambda item: np.linalg.norm(np.asarray(item['position'])-HINT_VIEW))
            self.hinted.add(hint['facility_id'])
            logger.info(f"探索第 {step} 屏：目录记录的关闭机关 {hint['facility_id']} 应在 "
                        f"{np.round(hint['position']).astype(int).tolist()}，移过去用新鲜检测确认。")
            self.move_by(np.asarray(hint['position'], float)-HINT_VIEW, pans=HINT_PANS,
                         tolerance=WAYPOINT_TOLERANCE/2)
            return self.look(step)
        if seen.get('minimap') and self.chases < MINIMAP_CHASES and 'enemy' in self.want:
            self.chases += 1
            point = min(seen['minimap'], key=lambda item: np.linalg.norm(item-MINIMAP_VIEW))
            logger.info(f'探索第 {step} 屏：小地图提示画面外 {np.round(point).astype(int).tolist()} 附近有敌人，'
                        '先把镜头移过去看。')
            self.move_by(point-MINIMAP_VIEW, pans=HINT_PANS, tolerance=WAYPOINT_TOLERANCE/2)
            return self.look(step)
        return None

    def result(self, found, step):
        """记录并返回找到的目标。"""
        if found['state'] == 'enemy':
            logger.info(f"探索第 {step} 屏发现可选敌人 {len(found['targets'])} 个，交回主循环核对后前往。")
        elif found['state'] == 'switch':
            logger.info(f"探索第 {step} 屏发现关闭机关 {found['target']['position']}"
                        f"（{found['target']['confidence']:.2f}），交由原生寻路前往；开门须另行复核。")
        return dict(found, views=self.views, pans=self.pans, step=step, camera=self.camera.tolist())

    def run(self):
        """原地先看一屏，再按螺旋移动；返回 dict(state=entry/enemy/switch/exhausted, ...)。"""
        logger.info(f"开始探索：找{'、'.join({'enemy': '敌人', 'switch': '关闭机关'}[kind] for kind in self.want)}，"
                    f'{len(self.waypoints)-1} 个航点由近及远，时限 {self.seconds:.0f} 秒。')
        primary, secondary = self.want[:1], self.want[1:]
        noted = None
        for step, waypoint in enumerate(self.waypoints):
            self.session.check()
            if time.monotonic()-self.started > self.seconds:
                logger.info(f'探索达到时限 {self.seconds:.0f} 秒，停在第 {step} 个航点。')
                break
            if step:
                self.move_by(waypoint-self.camera)
            if not self.fresh_view():
                continue
            seen = self.look(step)
            while seen['state'] != 'entry':
                found = self.pick(seen, primary)
                if found is not None:
                    return self.result(found, step)
                if noted is None and self.pick(seen, secondary) is not None:
                    noted = (step, self.camera.copy())
                    logger.info(f'探索第 {step} 屏记下{"敌人" if seen.get("enemies") else "关闭机关"}，'
                                '先看完其余航点找更优先的目标。')
                chased = self.chase(seen, step)
                if chased is None:
                    break
                seen = chased
            if seen['state'] == 'entry':
                return dict(state='entry', field=seen['field'], views=self.views, pans=self.pans)
        if noted is not None:
            step, camera = noted
            self.move_by(camera-self.camera)
            seen = self.look(step)
            if seen['state'] == 'entry':
                return dict(state='entry', field=seen['field'], views=self.views, pans=self.pans)
            found = self.pick(seen, secondary)
            if found is not None:
                return self.result(found, step)
            logger.info(f'倒回第 {step} 屏未再确认到记下的目标。')
        logger.info(f'探索结束未找到可用目标：{self.views} 屏，{self.pans} 次镜头移动，'
                    f'镜头离起点 {np.round(self.camera).astype(int).tolist()}，边界 {self.edges} 次，'
                    f'位移未测得 {self.unmeasured} 次，目录无法重定位 {self.catalog_errors} 次。')
        return dict(state='exhausted', field=self.runner.map_frame(track=False), views=self.views,
                    pans=self.pans, edges=self.edges, catalog_errors=self.catalog_errors,
                    duration=round(time.monotonic()-self.started, 1))
