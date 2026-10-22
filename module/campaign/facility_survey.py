"""从右上镜头边界逐行登记设施；敌人只作观察，不提前结束设施巡检。"""

import time

import numpy as np

from module.campaign.facility_memory import FacilityUnconfirmed, facility_frame, stable_facilities
from module.campaign.navigation import controls as goto
from module.campaign.sweep import ChapterSweep, HORIZONTAL_STEP, VERTICAL_STEP, SweepUnconfirmed, camera_motion
from module.campaign.vision.objects import detect_objects
from module.logger import logger


class FacilitySurvey(ChapterSweep):
    """复用已有镜头避让与运动检验，独立记录设施及覆盖，不把扫过当作清怪或道路已打开。

    默认180秒，最多沿用240次镜头预算。没有真实小队箭头或离屏标记时立即停止拖镜头。
    任何配准、边界、超时失败都保存partial；完整覆盖仅表示巡视过的镜头范围，不保证零漏检。
    """

    def __init__(self, runner, catalog, seconds=180., wanted=None, first_off=False):
        """绑定当前实例目录；wanted复核指定历史设施，显式first_off只找当前未试的关闭机关。"""
        super().__init__(runner)
        self.catalog, self.seconds, self.wanted = catalog, min(240., max(1., seconds)), wanted
        self.first_off = bool(first_off)
        self.visible = []
        self.occlusion_checked = False
        self.proof = dict(state='partial', source='scene_scan', origin='camera_top_right',
                          rows=0, pans=0, views=0, facilities=[], detection_complete=False)

    def sample(self):
        """连续检测机关、平台和屏障并登记几何身份，出现真实备战弹窗时立即交回战斗流程。"""
        self.session.check()
        if time.monotonic()-self.started >= self.seconds:
            raise SweepUnconfirmed('设施巡检时间预算已到，保存部分覆盖，不继续拖动。')
        field = self.runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            self.found = dict(state='entry')
            return None
        first = detect_objects(field, ('ground_mechanism', 'elevator', 'barrier'))
        self.session.pause(.25)
        field = self.runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            self.found = dict(state='entry')
            return None
        second = detect_objects(field, ('ground_mechanism', 'elevator', 'barrier'))
        stable = stable_facilities(first, second)
        self.visible = self.catalog.observe(field, stable)
        self.proof['views'] += 1
        self.proof['facilities'] = sorted(self.catalog.data['facilities'])
        self.proof['unresolved'] = self.catalog.unresolved
        self.session.record_evidence('facility_survey_view', field, pans=self.steps, rows=self.rows,
                                     facilities=self.visible, detections=second, arrays=dict(field=field))
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

    def measure_motion(self, before, after, drag):
        """允许有几何支持的斜向边缘响应继续登记；父类始终记录其覆盖缺口，不授予全图许可。

        仍要求分布充分的背景内点、重叠和朝手势方向运动。区别仅在设施采集可以保留部分
        结果继续探索，章节全清判定继续调用严格默认检查。
        """
        return camera_motion(before, after, drag, allow_oblique=True)

    def pan_anchor(self, field):
        """原规则未确认小队时，每次巡检最多移除一次小地图遮挡，仍须两帧真实头像通过。"""
        try:
            return super().pan_anchor(field)
        except SweepUnconfirmed:
            if self.occlusion_checked:
                raise
            self.occlusion_checked = True
            from module.campaign.vision.minimap_occlusion import unoccluded_offscreen
            self.session.begin_target('facility_avatar_occlusion')
            marker = unoccluded_offscreen(self.session)
            self.proof['unoccluded_avatar'] = marker
            if marker is None:
                raise SweepUnconfirmed('移除小地图遮挡后仍未双帧确认头像，未继续拖镜头。')
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
