"""受阻后从小队附近向外螺旋移动镜头，寻找新鲜双帧确认的关闭地面机关。

实录 2026-10-21（21-15）：光幕挡住主线敌人，门旁平台是单向到达端，点两次都不传送；
真正的开关在同一连通区南侧崖尖，镜头向南约四屏、向东两屏才进入画面，点一次后游戏
原生寻路走过去踩下，镜头随即切到原门且光幕消失。原有±360/240四次换视角看不到它。

实录 2026-10-21（21-32）：首领所在高台东口被光幕封住，光幕外站着未清的 21-20（灰色机器人
脚圈，置信度约0.71）；原搜索只找机关，镜头扫过它也不理会。因此同时找任务菱形下方以外的
两帧稳定敌人脚圈：守关敌人是真实关卡，开战仍须备战标题复核，不把找到当作开门。

镜头移动只用于观察；找到后由调用方点击并按实际结果复核。目录记录尽力而为，目录无法
重定位时不阻止本次新鲜检测，也不以新编号覆盖旧设施的尝试历史。
"""

import time

import numpy as np

from module.campaign.facility_memory import FacilityUnconfirmed, voted_facilities
from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.vision.enemy_ring import scene_ring_candidates
from module.campaign.vision.objects import detect_objects
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.story_marker import story_markers
from module.logger import logger

# 拖动手势向量（手指方向，镜头反向移动）；实测600px竖拖约半屏，边界处会被游戏夹住。
HORIZONTAL_DRAG = 700.
VERTICAL_DRAG = 520.
# 任务菱形正下方这一带的脚圈就是受阻的主线目标本身，不当作守关敌人。
QUEST_RING_DX = 120.
QUEST_RING_DY = 700.
# 每屏取帧数与投票门槛：烟雾会让单帧漏检（实录21-32三帧漏一帧）。先取两帧，两帧设施一致即
# 结束；不一致才补到四帧投票（实测 2026-10-22：固定四帧加逐帧配准每屏约15秒，识别本身约3秒）。
FACILITY_QUICK_FRAMES = 2
FACILITY_FRAMES = 4
FACILITY_VOTES = 2
# 场内主线菱形：蓝底提示样式识别为cue（实录22-27），同样标出受阻目标；edge是屏幕边缘的离屏指示。
QUEST_MARKER_KINDS = ('scene', 'cue')
# 原地第一屏中，刚撞墙目标附近的脚圈与它同在障碍后（实录21-32首领两侧普通敌人）。
BLOCKED_RADIUS = 300.
FACILITY_KINDS = ('ground_mechanism', 'elevator', 'barrier')


def spiral_drags(rings=4):
    """返回以起点为中心逐圈外扩的手势序列：先南、东，再北、西，每两段边长加一。"""
    directions = ((0., -VERTICAL_DRAG), (-HORIZONTAL_DRAG, 0.), (0., VERTICAL_DRAG), (HORIZONTAL_DRAG, 0.))
    drags, length = [], 1
    for leg in range(rings*2):
        drags += [directions[leg % 4]]*length
        if leg % 2:
            length += 1
    return [np.asarray(drag) for drag in drags]


def off_switches(frames):
    """多帧投票同位置、完整入屏、无同类竞争的关闭地面机关；开启机关和平台不在此列。"""
    return [item for item in voted_facilities(frames, FACILITY_VOTES)
            if item['kind'] == 'ground_mechanism' and item['state'] == 'off']


def frames_agree(frames):
    """两帧的完整入屏设施逐一对上（同类同状态≤12px），无漏检或闪现，不必再补帧投票。"""
    single = [len(voted_facilities([frame], 1)) for frame in frames]
    return len(set(single)) == 1 and len(voted_facilities(frames, len(frames))) == single[0]


def ring_centers(image):
    """本帧可点击敌人脚圈的框中心；覆盖小队落点的歧义框已由识别层剔除。"""
    rings, _ = scene_ring_candidates(image)
    return [(np.asarray([(item.box[0]+item.box[2])/2., (item.box[1]+item.box[3])/2.]), item.confidence)
            for item in rings]


def gate_enemies(first, second, markers):
    """两帧同位置（≤20px）且不在任务菱形正下方的敌人脚圈；返回第二帧中心与置信度。"""
    quest = [np.asarray(item['position'], float) for item in markers if item['kind'] in QUEST_MARKER_KINDS]
    found = []
    for center, confidence in second:
        if not any(np.linalg.norm(center-old) <= 20. for old, _ in first):
            continue
        if any(abs(center[0]-mark[0]) <= QUEST_RING_DX and 0. < center[1]-mark[1] <= QUEST_RING_DY
               for mark in quest):
            continue
        found.append((center, confidence))
    return found


class SwitchSearch:
    """有界螺旋镜头搜索；任何真实备战弹窗立即交回战斗流程，不把找到当作开门。

    机关优先、敌人兜底（用户 2026-10-22）：连续看完全部屏幕，途中见到关闭机关立即返回；
    敌人只记下屏号，全部屏幕都没有机关才把镜头倒回最早那一屏重新确认后交回。
    实录 2026-10-22（22-27）：螺旋第1、3、4屏的敌人都是光幕后的首领组，每次被打断去走一趟
    停滞约30秒，西侧开关所在的第6屏在5分钟内一直没看到。
    """

    def __init__(self, runner, catalog=None, pans=20, seconds=180., skip=(), enemies=True, blocked=(),
                 start=0, enemy_from=0, continuous=True):
        """skip是本受阻点已尝试过且未解决的搜索键，避免同状态下重复点同一个机关；
        enemies为False时只找关闭机关（本受阻点守关敌人尝试已达上限）；
        blocked是刚停滞目标在原地画面中的落点，原地第一屏不把它及附近脚圈当守关敌人；
        enemy_from之前各屏的敌人已试过走不到，不再记录；
        continuous为False表示本受阻点已完整看过全部屏幕的机关，只需找下一个敌人：
        start之前各屏只重放镜头移动、不取帧，见到敌人立即返回。"""
        self.runner, self.session, self.catalog = runner, runner.session, catalog
        self.enemies, self.enemy_from, self.continuous = enemies, enemy_from, continuous
        self.blocked = [np.asarray(point, float) for point in blocked]
        self.drags = spiral_drags()[:pans]
        self.seconds, self.skip, self.start = seconds, set(skip), start
        self.views, self.pans, self.catalog_errors = 0, 0, 0
        self.at = 0

    def look(self, step, goal=False):
        """取两帧（不一致补到四帧）检测关闭机关并登记目录；返回(状态, 候选, 新鲜帧)。

        goal为真表示镜头已由任务定位移到受阻目标旁：此处敌人多与目标同在障碍后，不找守关敌人；
        候选按离任务菱形最近排序（没有菱形时按画面中心），而不是按小队位置。
        同屏既有可用关闭机关又有敌人时先返回机关。
        """
        started = time.monotonic()
        field = self.runner.map_frame(track=False)
        if goto.battle_popup_score(field) > .8:
            return 'entry', None, field
        frames = [detect_objects(field, FACILITY_KINDS)]
        first_rings = ring_centers(field)
        fresh = field
        while len(frames) < FACILITY_FRAMES:
            if len(frames) >= FACILITY_QUICK_FRAMES and frames_agree(frames):
                break
            self.session.pause(.25)
            fresh = self.runner.map_frame(track=False)
            if goto.battle_popup_score(fresh) > .8:
                return 'entry', None, fresh
            frames.append(detect_objects(fresh, FACILITY_KINDS))
        second = frames[-1]
        enemies = []
        hunt = self.enemies and not goal and (step == 'goal' or step >= self.enemy_from)
        for center, confidence in (gate_enemies(first_rings, ring_centers(fresh), story_markers(fresh))
                                   if hunt else ()):
            # 实录 2026-10-21（21-32）：原地第一屏选中的“守关敌人”就是刚撞墙的同一个目标。
            if not step and any(np.linalg.norm(center-point) <= BLOCKED_RADIUS for point in self.blocked):
                continue
            key = ('enemy', step, *np.round(center/80.).astype(int).tolist())
            if key not in self.skip:
                enemies.append(dict(kind='enemy_ring', state='pending', position=center.tolist(),
                                    confidence=float(confidence), search_key=key))
        self.views += 1
        registered = []
        if self.catalog is not None:
            try:
                registered = self.catalog.observe(fresh, voted_facilities(frames, FACILITY_VOTES))
            except FacilityUnconfirmed:
                self.catalog_errors += 1
        candidates = []
        for item in off_switches(frames):
            key = (step, *np.round(np.asarray(item['position'])/80.).astype(int).tolist())
            known = [entry['facility_id'] for entry in registered if entry['kind'] == item['kind']
                     and np.linalg.norm(np.asarray(entry['position'])-item['position']) < 1.]
            if key in self.skip or (known and not self.catalog.eligible(
                    known[0], 'quest:'+str(self.runner.quest))):
                continue
            candidates.append(dict(item, search_key=key, **({'facility_id': known[0]} if known else {})))
        self.session.record_evidence('switch_search_view', fresh, step=step, detections=second,
                                     candidates=candidates, enemies=enemies, frames=len(frames),
                                     elapsed=round(time.monotonic()-started, 2), arrays=dict(field=fresh))
        self.session.preview(fresh)
        state, pool = ('found', candidates) if candidates else ('enemy', enemies)
        if not pool:
            return 'none', None, fresh
        if goal:
            marks = [item['position'] for item in story_markers(fresh) if item['kind'] in QUEST_MARKER_KINDS]
            anchor = np.asarray(marks[0] if len(marks) == 1 else goto.VIEWPORT_PIXEL, float)
        else:
            arrow = squad_arrow(fresh)
            anchor = np.asarray(goto.VIEWPORT_PIXEL if arrow is None else arrow, float)
        return state, min(pool, key=lambda item: np.linalg.norm(
            np.asarray(item['position'])-anchor)), fresh

    def pan(self, drag):
        """发送一次观察用镜头手势；小队落点保护同原螺旋。"""
        field = self.runner.map_frame(track=False)
        arrow = squad_arrow(field)
        ground = None if arrow is None else np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET]
        self.session.begin_target('switch_search_pan')
        pan_scene(self.session, drag, squad_ground=ground)
        self.pans += 1

    def revisit(self, step):
        """把镜头按原手势逆序倒回第step屏，重新取帧确认敌人；不沿用旧帧坐标。"""
        for index in range(self.at, step, -1):
            self.session.check()
            self.pan(-self.drags[index-1])
        self.at = step
        state, target, field = self.look(step)
        if state in ('entry', 'found', 'enemy'):
            return state, target, field
        return 'none', None, field

    def report(self, state, target, field, step):
        """记录并返回找到的机关或敌人。"""
        what = '关闭机关' if state == 'found' else '未清敌人脚圈'
        logger.info(f"螺旋搜索第 {step} 屏发现{what} {target['position']}"
                    f"（{target['confidence']:.2f}），交由原生寻路前往；开门或开战须另行复核。")
        return dict(state=state, target=target, field=field, views=self.views, pans=self.pans, step=step)

    def run(self):
        """原地先看一屏，再按螺旋逐步拖镜头；返回dict(state=found/enemy/entry/exhausted, ...)。

        complete为真表示全部屏幕都已看过机关（未被时间截断），调用方据此不再重复整圈找机关。
        """
        started = time.monotonic()
        logger.info(f'目标受阻：开始螺旋寻找关闭机关，最多 {len(self.drags)} 次镜头移动'
                    + (f'，前 {self.start} 屏已看过只移动镜头。' if self.start else '。'))
        noted, complete = None, True
        for step in range(len(self.drags)+1):
            self.session.check()
            if time.monotonic()-started > self.seconds:
                complete = False
                break
            if step:
                self.pan(self.drags[step-1])
                self.at = step
            # 实录 2026-10-22（22-27）：第3屏的敌人与目标同在光幕后，打不到；每次停滞后螺旋从第0屏
            # 重来，又在第3屏选中同一批敌人，三轮都没走到西侧开关所在的第6屏。
            if step < self.start:
                continue
            state, target, field = self.look(step)
            if state == 'entry':
                return dict(state='entry', field=field, views=self.views, pans=self.pans)
            if state == 'found' or (state == 'enemy' and not self.continuous):
                return dict(self.report(state, target, field, step), complete=False)
            if state == 'enemy' and noted is None:
                noted = step
                logger.info(f"螺旋搜索第 {step} 屏记下未清敌人脚圈 {target['position']}，先看完全部屏幕找关闭机关。")
        if noted is not None:
            state, target, field = self.revisit(noted)
            if state == 'entry':
                return dict(state='entry', field=field, views=self.views, pans=self.pans)
            if state != 'none':
                return dict(self.report(state, target, field, noted), complete=complete)
            logger.info(f'倒回第 {noted} 屏未再确认到敌人脚圈。')
        logger.info(f'螺旋搜索未发现可用关闭机关：{self.views} 屏，{self.pans} 次镜头移动，'
                    f'目录无法重定位 {self.catalog_errors} 次。')
        return dict(state='exhausted', views=self.views, pans=self.pans, catalog_errors=self.catalog_errors,
                    complete=complete)
