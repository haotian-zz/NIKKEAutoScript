"""无全章底图依赖的蛇形镜头巡检；只在有运动和边界证据时确认扫描范围。"""

import time

import cv2
import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.scene_memory import ring_frame, stable_candidates
from module.campaign.vision.detection import detect_minimap
from module.campaign.vision.enemy_ring import scene_registration
from module.campaign.vision.ex_ring import minimap_enemy_hints
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.minimap import map_visible
from module.campaign.vision.perception import minimap_squad_ground, squad_arrow
from module.campaign.vision.surface_motion import project
from module.logger import logger

SWEEP_LIMIT = 240
HORIZONTAL_STEP = 480.
VERTICAL_STEP = 260.
EDGE_CONFIRMATIONS = 2
# 确认离屏后镜头往回（朝小队方向）累计移动超过这么多像素，就不再沿用离屏结论，重新识别。
SQUAD_RETURN = 40.
# 按累计位移推算的小队落点超出客户区这么多像素才算离屏（主平移有视差误差）。
SQUAD_OFFSCREEN_MARGIN = 120.
# 一轮巡检最多把几次配准失败当作雾边边界；超过说明不是地图边缘，按原样停止。
FOG_EDGE_LIMIT = 6


class SweepUnconfirmed(RuntimeError):
    """巡检缺乏运动或边界证据；禁止把未检查区域当成没有敌人。"""
    reason = 'chapter_clearance_unconfirmed'


class SweepMotionLost(SweepUnconfirmed):
    """拖动后严格配准和主平移都测不到镜头位移，常见于拖进地图外的雾或空白。"""


def camera_motion(before, after, drag, allow_oblique=False):
    """在背景几何内点范围内估计镜头响应，拒绝反向、过大或大幅横向漂移的未知变换。

    返回客户区内容位移向量。小于 3px 才能作为单次边界候选；调用方须连续验证两次。
    网格至少六点处于双帧内点凸包，横纵跨度各 200px、凸包面积 60000px²，避免固定采样
    恰好落在地图外空白或仅有单行支持。位移上限保留至少约一半视野重叠，不以大幅跳图连接覆盖。
    allow_oblique仅供保留partial的设施探索，允许边缘约束下的斜向响应；不能据此确认矩形条带覆盖。
    """
    try:
        matrix, source_hull, destination_hull = scene_registration(before, after)
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        raise SweepUnconfirmed(f'镜头背景配准未通过，未确认巡检覆盖：{exc}') from exc
    points = np.array([(x, y) for y in np.linspace(200., 820., 5) for x in np.linspace(250., 1490., 7)])
    supported = np.array([cv2.pointPolygonTest(source_hull, tuple(point), False) >= 0 for point in points])
    mapped = project(matrix, points)
    supported &= [cv2.pointPolygonTest(destination_hull, tuple(point), False) >= 0 for point in mapped]
    support = points[supported].astype(np.float32)
    if (len(support) < 6 or min(np.ptp(support, axis=0)) < 200.
            or abs(cv2.contourArea(cv2.convexHull(support))) < 60000.):
        raise SweepUnconfirmed('镜头运动缺少分布充分的背景内点，未确认巡检覆盖。')
    shifts = mapped[supported] - points[supported]
    delta = np.median(shifts, axis=0)
    corners = np.float32([[250., 120.], [1490., 120.], [1490., 880.], [250., 880.]])
    denominators = np.c_[corners, np.ones(4)] @ matrix[2]
    if min(denominators) <= 0 or max(denominators) / min(denominators) > 3.:
        raise SweepUnconfirmed('相机投影接近地平线，不能连接巡检覆盖。')
    footprint = project(np.linalg.inv(matrix), corners).astype(np.float32)
    area = abs(cv2.contourArea(corners))
    intersection, _ = cv2.intersectConvexConvex(corners, footprint)
    if intersection / area < .45 or not .4 <= abs(cv2.contourArea(footprint)) / area <= 2.5:
        raise SweepUnconfirmed('相邻巡检视野重叠不足，不能连接全章覆盖。')
    if np.max(np.linalg.norm(shifts, axis=1)) < 3.:
        return np.zeros(2)
    axis = int(np.argmax(np.abs(drag)))
    if delta @ np.asarray(drag) <= 0 or (not allow_oblique and abs(delta[1 - axis]) > 80.):
        raise SweepUnconfirmed(f'镜头响应与巡检手势不一致：手势 {list(drag)}，'
                               f'实际位移 {np.round(delta, 2).tolist()}。')
    return delta


class ChapterSweep:
    """从镜头边界开始逐行扫描，发现敌人立即交回战斗；最后一场胜利后重新完整巡检。

    每一屏用两帧红圈和小地图标记复核。左右及上下边界均需实测，未识别、受阻或 EX 标记
    均阻止清空确认。日志保存每次巡检证据，坐标仅属于当前游戏画面，不使用章节编号特判。
    """

    def __init__(self, runner):
        """绑定已有会话和观察簿；本次巡检的边界及待处理标记不跨战斗复用。"""
        self.runner, self.session, self.memory = runner, runner.session, runner.memory
        self.steps, self.rows, self.hints, self.axes = 0, 0, [], set()
        self.coverage_warnings = []
        # 巡检开始以来画面内容的累计实测位移；确认小队离屏时记下离屏方向和当时的累计位移。
        self.content, self.squad_away, self.occlusion_checked = np.zeros(2), None, False
        # 最后一次看到小队箭头时的地面点与当时的累计位移。
        self.squad_seen = None
        self.started = time.monotonic()
        self.found = None
        self.fog_edges = 0
        self.proof = dict(state='incomplete', rows=0, pans=0, pending=[])

    def sample(self):
        """连续观测同一视野的全部红圈与小地图普通／EX 标记，不因不可点击而忽略剩余敌人。"""
        field = self.runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            self.found = dict(state='entry')
            return None
        first_frame = ring_frame(field, self.runner.owner.chapter_ocr)
        before_field = field
        first = self.memory.observe(field, frame=first_frame)
        self.session.pause(.25)
        field = self.runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            self.found = dict(state='entry')
            return None
        frame = ring_frame(field, self.runner.owner.chapter_ocr)
        second = self.memory.observe(field, frame=frame)
        candidates = [item for item in stable_candidates(first, second) if self.runner.selectable(item)]
        panel = cv2.resize(field, (1920, 1080))[96:307, 25:243]
        if not map_visible(panel):
            raise SweepUnconfirmed('紧凑小地图未确认可见，不能把缺失的敌人标记当作零个。')
        markers = [dict(kind=item.label, position=np.asarray(item.center).tolist(), confidence=item.confidence)
                   for item in detect_minimap(panel, ('minimap_enemy_normal', 'minimap_enemy_ex'))]
        markers.extend(minimap_enemy_hints(panel))
        if second or markers:
            self.hints.append(dict(view=self.steps, rings=len(second), minimap=markers))
        # 原始像素保留在有界证据槽中；JPEG 不能复现细小文字与发光圈的颜色阈值。
        if (first or second) and not candidates:
            arrays = dict(field=field, previous_field=before_field)
        else:
            arrays = dict(field=field) if markers and not second else None
        self.session.record_evidence('chapter_sweep_view', field, pans=self.steps, rows=self.rows,
                                     remaining=second, previous_enemies=first, stable=len(candidates),
                                     minimap=markers, arrays=arrays)
        logger.info(f'清怪巡检第 {self.steps} 屏：场景目标 {len(second)} 个，'
                    f'可选 {len(candidates)} 个，小地图待处理提示 {len(markers)} 个。')
        if candidates:
            self.found = dict(state='target', field=field, targets=candidates)
        return dict(field=field, frame=frame)

    def move(self, drag, before=None):
        """发送一次避开小队的相机拖动并验证实际响应；输入失败保留失败，不伪装成边界。"""
        if self.steps >= SWEEP_LIMIT:
            raise SweepUnconfirmed(f'本轮巡检达到 {SWEEP_LIMIT} 次镜头动作，尚未确认全章覆盖。')
        before = self.sample() if before is None else before
        if self.found is not None:
            return None
        field = before['field']
        ground = self.pan_anchor(field)
        self.session.begin_target('chapter_sweep_pan')
        self.pan(np.asarray(drag, float), ground)
        self.steps += 1
        after = self.sample()
        if self.found is not None:
            return None
        delta = self.measure_motion(before['frame'], after['frame'], drag)
        axis = int(np.argmax(np.abs(drag)))
        if abs(delta[axis]) >= 3.:
            self.axes.add(axis)
        if abs(delta[1 - axis]) > max(25., .2 * abs(delta[axis])):
            self.coverage_warnings.append(dict(pan=self.steps, drag=list(drag), actual=delta.tolist()))
            logger.info('相机边缘存在斜向响应，继续有界探索；本轮覆盖保留存疑，不允许报全清。')
        self.session.record_evidence('chapter_sweep_motion', after['field'], drag=drag,
                                     actual=delta.tolist(), pans=self.steps, rows=self.rows)
        return delta

    def pan(self, drag, ground):
        """发送一次避开小队的镜头手势，固定等待画面稳定。"""
        pan_scene(self.session, drag, squad_ground=ground)

    def measure_motion(self, before, after, drag):
        """清怪扫描使用严格轴向响应；严格配准失败时按主平移继续找敌人，覆盖记为存疑。

        实录 2026-10-22（22-36）：高架视差让第 1 屏就报“缺少分布充分的背景内点”，整轮中止。
        主平移只用来继续巡检找剩余敌人；有存疑记录时 run() 不会声明本章全清。
        """
        try:
            delta = camera_motion(before, after, drag)
        except SweepUnconfirmed as exc:
            from module.campaign.switch_search import content_shift
            shift = content_shift(None, None, drag, before, after)
            if shift is None:
                raise SweepMotionLost(str(exc)) from exc
            self.coverage_warnings.append(dict(pan=self.steps, drag=list(drag), actual=shift.tolist(),
                                               reason=str(exc)))
            logger.info(f'镜头严格配准未通过，按主平移 {np.round(shift).astype(int).tolist()} 继续巡检；覆盖记为存疑。')
            delta = shift
        self.content = self.content + delta
        return delta

    def pan_anchor(self, field):
        """真实箭头或离屏标记允许拖镜头；确认离屏后镜头继续远离小队时沿用离屏结论。

        实录 2026-10-22（22-36）：设施巡检与清怪巡检都遇到过——镜头巡到地图边缘，小队离屏头像在左侧正好被小地图挡住；
        每次巡检只允许一次收起小地图复核，第二次就停了。离屏方向和累计实测位移足以判断：
        画面内容沿离屏方向继续移动，小队只会更远，不可能落到拖动按下点上。
        """
        arrow = squad_arrow(field)
        if arrow is not None:
            ground = np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET]
            self.squad_away, self.squad_seen = None, (ground, self.content.copy())
            return ground
        ground = minimap_squad_ground(field)
        if ground is not None:
            self.squad_away, self.squad_seen = None, (ground, self.content.copy())
            return ground
        marker = squad_offscreen(field)
        if marker is None and self.squad_seen is not None:
            # 实录 2026-10-22（22-36）：清怪巡检拖了四屏后小队在右上角穹顶里，箭头贴着任务栏被 HUD 排除，
            # 也没有离屏头像。小队巡检期间不动，按看到时的位置加累计实测位移推算。
            ground, content = self.squad_seen
            projected = ground+self.content-content
            height, width = field.shape[:2]
            margin = SQUAD_OFFSCREEN_MARGIN
            if not (-margin <= projected[0] <= width+margin and -margin <= projected[1] <= height+margin):
                return None
            logger.info(f'未识别到小队箭头，按累计位移推算小队在 {np.round(projected).astype(int).tolist()}，拖动时避开。')
            return projected
        if marker is None and self.squad_away is not None:
            direction, content = self.squad_away
            if (self.content-content) @ direction >= -SQUAD_RETURN:
                return None
        if marker is None and not self.occlusion_checked:
            self.occlusion_checked = True
            from module.campaign.vision.minimap_occlusion import unoccluded_offscreen
            self.session.begin_target('sweep_avatar_occlusion')
            marker = unoccluded_offscreen(self.session)
            self.proof['unoccluded_avatar'] = marker
        if marker is None:
            raise SweepUnconfirmed('未确认小队位置或离屏标记，停止镜头巡检以避免发送走位。')
        angle = np.radians(marker['direction'])
        self.squad_away = (np.array([np.cos(angle), np.sin(angle)]), self.content.copy())
        return None

    def fog_edge(self, drag, exc):
        """拖进雾里测不到位移时拖回一格，把这个方向当作边界；覆盖记为存疑，不允许报全清。

        实录 2026-10-25（HARD 9）：左侧是悬崖外的大片雾，找左边界时镜头拖进雾里配准失败，
        连续三轮整轮中止，一行都没往下扫，小屋下方离小队一两屏的敌人始终没进过画面。
        """
        self.fog_edges += 1
        if self.fog_edges > FOG_EDGE_LIMIT:
            raise SweepUnconfirmed(f'巡检已 {FOG_EDGE_LIMIT} 次在雾边测不到镜头位移，停止：{exc}') from exc
        field = self.runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            self.found = dict(state='entry')
            return
        back = -np.asarray(drag, float)
        self.session.begin_target('chapter_sweep_fog_back')
        self.pan(back, self.pan_anchor(field))
        self.steps += 1
        self.coverage_warnings.append(dict(pan=self.steps, drag=list(drag), reason=f'fog_edge: {exc}'))
        self.session.record_evidence('chapter_sweep_fog_edge', self.runner.map_frame(), drag=list(drag),
                                     pans=self.steps, rows=self.rows, reason=str(exc))
        logger.info(f'镜头拖进雾里测不到位移，拖回一格并把 {list(drag)} 方向当作边界；覆盖记为存疑。')

    def edge(self, drag):
        """沿一轴走到连续两次几何静止的边界；新敌人、取消与输入故障立即结束当前巡检。"""
        still = 0
        for _ in range(32):
            try:
                delta = self.move(drag)
            except SweepMotionLost as exc:
                self.fog_edge(drag, exc)
                return self.found is None
            if self.found is not None:
                return False
            still = still + 1 if np.linalg.norm(delta) < 3. else 0
            if still >= EDGE_CONFIRMATIONS:
                return True
        raise SweepUnconfirmed('单轴巡检未能确认边界，不能声明扫描完整。')

    def run(self):
        """先到左上边界，再蛇形逐行到最下边界；零敌人且两轴确实响应过才允许清空确认。

        输入或几何失败向上传播。扫描遇敌只返回当前两帧候选，战斗后必须重新开始完整巡检。
        """
        logger.info('开始全章清怪巡检：确认镜头边界并逐行扫描普通怪、分支及 EX 标记。')
        try:
            if not self.edge([HORIZONTAL_STEP, 0.]) or not self.edge([0., VERTICAL_STEP]):
                return self.found
            direction = -1
            for _ in range(64):
                self.rows += 1
                if not self.edge([direction * HORIZONTAL_STEP, 0.]):
                    return self.found
                try:
                    delta = self.move([0., -VERTICAL_STEP])
                except SweepMotionLost as exc:
                    self.fog_edge([0., -VERTICAL_STEP], exc)
                    if self.found is not None:
                        return self.found
                    break
                if self.found is not None:
                    return self.found
                if np.linalg.norm(delta) < 3.:
                    confirmation = self.move([0., -VERTICAL_STEP])
                    if self.found is not None:
                        return self.found
                    if np.linalg.norm(confirmation) < 3.:
                        break
                direction *= -1
            else:
                raise SweepUnconfirmed('巡检未能确认地图下边界。')
            if self.axes != {0, 1}:
                raise SweepUnconfirmed('两轴镜头未分别获得运动证据，静止画面不能证明全章边界。')
            if self.coverage_warnings:
                raise SweepUnconfirmed('巡检存在斜向相机响应或雾边测不到位移，已保留覆盖存疑证据，不能声明本章全清。')
            if self.hints:
                self.proof['pending'] = self.hints
                raise SweepUnconfirmed('巡检仍发现红圈或小地图敌人标记，清怪未完成。')
            self.proof.update(state='verified_empty', rows=self.rows, pans=self.steps,
                              duration=round(time.monotonic() - self.started, 2))
            self.session.record_evidence('chapter_clearance', self.session.win.client(), **self.proof)
            logger.info(f'全章清怪巡检通过：{self.rows} 行，{self.steps} 次镜头动作，未发现剩余敌人标记。')
            return dict(state='verified_empty', proof=self.proof)
        finally:
            self.proof.update(rows=self.rows, pans=self.steps, pending=self.hints,
                              unconfirmed_motion=self.coverage_warnings)
            self.session.evidence.save(self.session.evidence.folder / 'chapter-clearance', self.proof)
