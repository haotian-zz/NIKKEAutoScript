"""从右上镜头边界逐行登记设施；敌人只作观察，不提前结束设施巡检。"""

import time

import numpy as np

from module.campaign.facility_memory import FacilityUnconfirmed, facility_frame, voted_facilities
from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.navigation.facility_map import record_map_positions
from module.campaign.sweep import ChapterSweep, HORIZONTAL_STEP, VERTICAL_STEP, SweepUnconfirmed, camera_motion
from module.campaign.vision.objects import detect_objects
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.story_marker import story_markers
from module.logger import logger

# 确认离屏后镜头往回（朝小队方向）累计移动超过这么多像素，就不再沿用离屏结论，重新识别。
SQUAD_RETURN = 40.


class FacilitySurvey(ChapterSweep):
    """复用已有镜头避让与运动检验，独立记录设施及覆盖，不把扫过当作清怪或道路已打开。

    默认180秒，最多沿用240次镜头预算。没有真实小队箭头或离屏标记时立即停止拖镜头。
    任何配准、边界、超时失败都保存partial；完整覆盖仅表示巡视过的镜头范围，不保证零漏检。
    """

    def __init__(self, runner, catalog, seconds=180., wanted=None, first_off=False):
        """绑定当前实例目录；wanted复核指定历史设施，显式first_off只找当前未试的关闭机关。"""
        super().__init__(runner)
        self.catalog, self.seconds, self.wanted = catalog, min(420., max(1., seconds)), wanted
        self.first_off = bool(first_off)
        self.visible = []
        self.occlusion_checked = False
        # 巡检开始以来画面内容的累计实测位移；确认小队离屏时记下离屏方向和当时的累计位移。
        self.content = np.zeros(2)
        self.squad_away = None
        self.proof = dict(state='partial', source='scene_scan', origin='camera_top_right',
                          rows=0, pans=0, views=0, facilities=[], detection_complete=False)

    def sample(self):
        """连续检测机关、平台和屏障并登记几何身份，出现真实备战弹窗时立即交回战斗流程。

        取帧不做背景桥接：每屏都由本方法登记一次目录。实录 2026-10-22（22-36）：取帧顺带的桥接
        每次都要和全部关键帧配准，开局巡检每屏耗时约13秒，还多塞桥接帧。
        """
        self.session.check()
        if time.monotonic()-self.started >= self.seconds:
            raise SweepUnconfirmed('设施巡检时间预算已到，保存部分覆盖，不继续拖动。')
        field = self.runner.map_frame(track=False)
        if goto.battle_popup_score(field) > .8:
            self.found = dict(state='entry')
            return None
        from module.campaign.switch_search import (FACILITY_FRAMES, FACILITY_KINDS, FACILITY_QUICK_FRAMES,
                                                   FACILITY_VOTES, frames_agree, ring_centers)
        frames = [detect_objects(field, FACILITY_KINDS)]
        while len(frames) < FACILITY_FRAMES:
            # 与螺旋相同：两帧设施一致即结束，不一致才补到四帧投票。
            if len(frames) >= FACILITY_QUICK_FRAMES and frames_agree(frames):
                break
            self.session.pause(.25)
            field = self.runner.map_frame(track=False)
            if goto.battle_popup_score(field) > .8:
                self.found = dict(state='entry')
                return None
            frames.append(detect_objects(field, FACILITY_KINDS))
        second = frames[-1]
        # 烟雾会让单帧漏检，按多帧投票登记；同屏敌人脚圈与任务标记一并记录，供全章概览。
        stable = voted_facilities(frames, FACILITY_VOTES)
        self.visible = self.catalog.observe(field, stable)
        moved = record_map_positions(self.runner, self.catalog, self.visible, field)
        if moved is not None:
            # 开合地图后以新帧作运动基准；画面若变了，本屏检测结果不再当作当前像素。
            field = moved['field']
            if not moved['unchanged']:
                self.visible, second = [], detect_objects(field, FACILITY_KINDS)
        enemies = [dict(position=center.tolist(), confidence=float(score)) for center, score in ring_centers(field)]
        markers = story_markers(field)
        self.proof['views'] += 1
        self.proof['facilities'] = sorted(self.catalog.data['facilities'])
        self.proof['unresolved'] = self.catalog.unresolved
        self.session.record_evidence('facility_survey_view', field, pans=self.steps, rows=self.rows,
                                     facilities=self.visible, detections=second, enemies=enemies,
                                     markers=markers, arrays=dict(field=field))
        self.session.preview(field)
        if self.first_off:
            candidates = [item for item in self.visible
                          if item['kind'] == 'ground_mechanism' and item['state'] == 'off'
                          and self.catalog.eligible(item['facility_id'], 'quest:'+str(self.runner.quest))]
            if candidates:
                self.found = dict(state='facility_visible', facility_id=candidates[0]['facility_id'],
                                  target=candidates[0], source='fresh_two_frame_scan')
        if self.wanted is not None and any(item['facility_id'] == self.wanted for item in self.visible):
            self.found = dict(state='facility_visible', facility_id=self.wanted)
        return dict(field=field, frame=facility_frame(field, second))

    def pan(self, drag, ground):
        """按时长推进拖动、画面静止即取帧（与螺旋相同，实测每屏省约2秒），不改变运动校验。"""
        pan_scene(self.session, drag, squad_ground=ground, settle=None, paced=True)

    def measure_motion(self, before, after, drag):
        """测出本次拖动的画面位移并累计，供小队离屏推算。"""
        delta = self.camera_shift(before, after, drag)
        self.content = self.content + delta
        return delta

    def camera_shift(self, before, after, drag):
        """允许有几何支持的斜向边缘响应继续登记；父类始终记录其覆盖缺口，不授予全图许可。

        仍要求分布充分的背景内点、重叠和朝手势方向运动。区别仅在设施采集可以保留部分
        结果继续探索，章节全清判定继续调用严格默认检查。
        """
        try:
            return camera_motion(before, after, drag, allow_oblique=True)
        except SweepUnconfirmed as exc:
            # 实录 2026-10-22（22-36）：高架与高楼视差让严格单应失败，开局巡检第二次拖动就中止。
            # 改用螺旋同款主平移测镜头位移继续登记；覆盖记为存疑，不会报全图完成。
            from module.campaign.switch_search import content_shift
            shift = content_shift(None, None, drag, before, after)
            if shift is None:
                raise
            self.coverage_warnings.append(dict(pan=self.steps, drag=list(drag), actual=shift.tolist(),
                                               reason=str(exc)))
            logger.info(f'镜头严格配准未通过，按主平移 {np.round(shift).astype(int).tolist()} 继续巡检；覆盖记为存疑。')
            return shift

    def pan_anchor(self, field):
        """真实箭头或离屏标记允许拖镜头；确认离屏后镜头继续远离小队时沿用离屏结论。

        实录 2026-10-22（22-36）：镜头巡到地图右缘，小队离屏头像在左侧正好被小地图挡住；
        每次巡检只允许一次收起小地图复核，第二次就停了。离屏方向和累计实测位移足以判断：
        画面内容沿离屏方向继续移动，小队只会更远，不可能落到拖动按下点上。
        """
        arrow = squad_arrow(field)
        if arrow is not None:
            self.squad_away = None
            return np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET]
        marker = squad_offscreen(field)
        if marker is None and self.squad_away is not None:
            direction, content = self.squad_away
            if (self.content-content) @ direction >= -SQUAD_RETURN:
                return None
        if marker is None and not self.occlusion_checked:
            self.occlusion_checked = True
            from module.campaign.vision.minimap_occlusion import unoccluded_offscreen
            self.session.begin_target('facility_avatar_occlusion')
            marker = unoccluded_offscreen(self.session)
            self.proof['unoccluded_avatar'] = marker
        if marker is None:
            raise SweepUnconfirmed('未确认小队位置或离屏标记，停止镜头巡检以避免发送走位。')
        angle = np.radians(marker['direction'])
        self.squad_away = (np.array([np.cos(angle), np.sin(angle)]), self.content.copy())
        return None

    def run(self):
        """先确认右、上边界，再左右交替向下巡视；下边界双帧静止和两轴运动通过才记coverage。

        斜向边缘响应意味着条带可能漏扫，因此保留partial。识别数量增加、拖镜头均不续期预算。
        目录保存静态关键帧和设施身份；敌人出现不改变本次扫描目的或触发自动开战。
        """
        logger.info('开始设施巡检：右上边界起始，逐屏登记机关、平台和屏障。')
        try:
            if not self.edge([-HORIZONTAL_STEP, 0.]) or not self.edge([0., VERTICAL_STEP]):
                return self.found
            direction = 1
            for _ in range(64):
                self.rows += 1
                if not self.edge([direction*HORIZONTAL_STEP, 0.]):
                    return self.found
                delta = self.move([0., -VERTICAL_STEP])
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
                raise SweepUnconfirmed('设施巡检未确认地图下边界。')
            if self.axes != {0, 1} or self.coverage_warnings:
                raise SweepUnconfirmed('设施巡检缺少两轴覆盖或存在斜向边缘响应，未标记全图完成。')
            self.proof['state'] = 'camera_coverage_verified'
            return dict(state='surveyed', proof=self.proof)
        except (SweepUnconfirmed, FacilityUnconfirmed) as exc:
            self.proof['reason'] = str(exc)
            raise
        finally:
            self.proof.update(rows=self.rows, pans=self.steps,
                              duration=round(time.monotonic()-self.started, 2),
                              unconfirmed_motion=self.coverage_warnings)
            self.catalog.data['survey'] = self.proof
            self.catalog.save()
            self.session.evidence.save(self.session.evidence.folder/'facility-survey', self.proof)
