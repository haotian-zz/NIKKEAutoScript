"""原生点击寻路的位移观测；仅测镜头与小队，不判断地面颜色或道路可达性。"""

import cv2
import numpy as np


class NativeMotion:
    """用双向光流估计背景变化，再扣除小队箭头的镜头位移。

    每次只比较相邻的新帧，输出moving/stationary/unknown。缺箭头、缺纹理、支持域不足
    或几何冲突均输出unknown；不能把观测失败当作停止，也不生成游戏输入。
    """

    def __init__(self, field, arrow=None, target=None):
        """保存点击前原图与箭头；target仅用于剔除敌人动画，不作为位移锚点。"""
        self.previous = (field.copy(), None if arrow is None else np.asarray(arrow, float))
        self.target = None if target is None else np.asarray(target, float)

    def observe(self, field, arrow):
        """客户区缩半后计算双向角点流；背景支持覆盖箭头时才报告真实残差。

        移动阈值为原客户区12px，静止阈值为4px，中间区间保留unknown以吸收呼吸动画。
        输出的位移不是到达证明，仍须实际备战或设施状态变化。
        """
        before, old = self.previous
        arrow = None if arrow is None else np.asarray(arrow, float)
        self.previous = (field.copy(), arrow)
        if (old is None or arrow is None or not np.isfinite([old, arrow]).all()
                or before.shape != (999, 1776, 3) or field.shape != before.shape):
            return dict(state='unknown', reason='squad_unobserved')
        gray0, gray1 = [cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (888, 500))
                        for image in (before, field)]
        scale = np.array([.5, 500/999])
        mask = np.zeros(gray0.shape, np.uint8)
        x, y = old * scale
        cv2.rectangle(mask, (max(123, int(x-200)), max(80, int(y-140))),
                      (min(744, int(x+200)), min(440, int(y+180))), 255, -1)
        cv2.rectangle(mask, (int(x-48), int(y-25)), (int(x+48), int(y+90)), 0, -1)
        if self.target is not None:
            tx, ty = self.target*scale
            cv2.rectangle(mask, (int(tx-45), int(ty-85)), (int(tx+45), int(ty+25)), 0, -1)
        points = cv2.goodFeaturesToTrack(gray0, 300, .02, 8, mask=mask, blockSize=5)
        if points is None or len(points) < 30:
            return dict(state='unknown', reason='background_unobserved')
        options = dict(winSize=(25, 25), maxLevel=3, criteria=(3, 40, .01))
        moved, valid, error = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None, **options)
        if moved is None:
            return dict(state='unknown', reason='flow_failed')
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, moved, None, **options)
        if back is None:
            return dict(state='unknown', reason='reverse_flow_failed')
        keep = ((valid[:, 0] > 0) & (reverse[:, 0] > 0) & (error[:, 0] < 18)
                & (np.linalg.norm(points[:, 0]-back[:, 0], axis=1) < .7))
        source, destination = points[keep, 0], moved[keep, 0]
        if len(source) < 30:
            return dict(state='unknown', reason='few_consistent_corners')
        matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                                     ransacReprojThreshold=1.5)
        if matrix is None or inliers is None:
            return dict(state='unknown', reason='background_fit_failed')
        inside = inliers[:, 0].astype(bool)
        if inside.sum() < 25 or inside.mean() < .65 or not np.isfinite(matrix).all():
            return dict(state='unknown', reason='background_fit_ambiguous')
        scale_change = np.linalg.norm(matrix[:2, 0])
        source_hull = cv2.convexHull(source[inside])
        destination_hull = cv2.convexHull(destination[inside])
        if (not .9 <= scale_change <= 1.1
                or cv2.pointPolygonTest(source_hull, tuple(old*scale), False) < 0
                or cv2.pointPolygonTest(destination_hull, tuple(arrow*scale), False) < 0):
            return dict(state='unknown', reason='squad_outside_background_support')
        residual = (arrow*scale - matrix @ np.r_[old*scale, 1.]) / scale
        distance = float(np.linalg.norm(residual))
        state = 'moving' if distance >= 12 else 'stationary' if distance <= 4 else 'unknown'
        return dict(state=state, reason='camera_compensated', movement_px=round(distance, 2),
                    movement_vector=np.round(residual, 2).tolist(), inliers=int(inside.sum()))


class MinimapStability:
    """以固定UI的小队圆心和非退化道路轮廓交叉确认静止，消除场景箭头上下动画。

    只做同屏相等性检查，不定位到底图或推断道路可达。小队圆心与道路须同时稳定；
    直路缺少转角、遮挡过多、漏圈或镜头变化均保持unknown。
    """

    def __init__(self, field, reader=None):
        """保存点击前的独立小地图观测；reader可注入纯图像替身，生产使用唯一圆环精定位。"""
        self.reader = reader or self.read
        self.reference = self.snapshot(field)

    @staticmethod
    def read(field):
        """确认紧凑面板控件后读取圆环、道路及有效域；不接受展开地图或其他页面。"""
        from module.campaign.vision.minimap import map_visible, minimap_control_visible
        from module.campaign.vision.perception import minimap_masks
        if (not minimap_control_visible(field[76:120, 20:64])
                or not map_visible(field[118:273, 27:207])):
            raise ValueError('Compact minimap not visible')
        _, road, valid, player = minimap_masks(field)
        return road, valid, np.asarray(player, float)

    def snapshot(self, field):
        """只将可辨识的二维轮廓纳入比较，避免长直路滚动但外观不变时误称静止。"""
        try:
            road, valid, player = self.reader(field)
        except ValueError:
            return None
        mask = cv2.erode(valid, np.ones((7, 7), np.uint8))
        points = cv2.goodFeaturesToTrack(road, 40, .03, 4, mask=mask)
        if (points is None or len(points) < 6 or min(np.ptp(points[:, 0], axis=0)) < 32
                or cv2.contourArea(cv2.convexHull(points)) < 1500):
            return None
        return road, valid, player

    def observe(self, field):
        """始终与窗口起始帧比较，拒绝把缓慢累计位移拆成多个小变化而误判静止。"""
        current, previous = self.snapshot(field), self.reference
        if current is None or previous is None:
            self.reference = current
            return dict(state='unknown', reason='minimap_unobserved')
        road, valid, player = current
        old_road, old_valid, old_player = previous
        common = (valid > 0) & (old_valid > 0)
        union = ((road > 0) | (old_road > 0)) & common
        if (np.count_nonzero(common) < max(2000, .65 * max(np.count_nonzero(valid), np.count_nonzero(old_valid)))
                or np.count_nonzero(union) < 200):
            self.reference = current
            return dict(state='unknown', reason='minimap_support_insufficient')
        shift = float(np.linalg.norm(player-old_player))
        changed = float(np.mean((road[union] > 0) != (old_road[union] > 0)))
        if shift > .75 or changed > .01:
            self.reference = current
            return dict(state='unknown', reason='minimap_changed')
        return dict(state='stationary', reason='independent_minimap_stable',
                    ring_shift_px=round(shift, 3), road_change=round(changed, 5))


def combine_motion(scene, minimap):
    """独立小地图可确认箭头动画下的静止；与可靠场景移动冲突时保留unknown。

    已标定箭头动画为竖直10～18px振幅；只有横向≤3px、纵向≤18px的场景残差可由
    稳定小地图消歧。更大位移不被小地图覆盖，避免以别名道路图案否认真实行走。
    """
    if minimap['state'] != 'stationary':
        return scene
    if scene['state'] == 'moving':
        delta = scene.get('movement_vector')
        if delta is None or abs(delta[0]) > 3 or abs(delta[1]) > 18:
            return dict(state='unknown', reason='motion_observations_disagree')
    return dict(minimap, scene_feedback=scene)


class ClickProgress:
    """有界跟踪点击后的结果，不以超时或无法观测直接宣称道路不可达。"""

    def __init__(self, started, unknown_limit=20., total_limit=60., grace=2., stable_seconds=2.):
        """初始响应留2秒，连续三次且跨2秒静止才报告stalled；真实移动最多跟随60秒。"""
        self.started, self.soft_deadline = started, started+unknown_limit
        self.unknown_limit, self.total_limit = unknown_limit, total_limit
        self.grace, self.stable_seconds = grace, stable_seconds
        self.stable_since, self.stable_count = None, 0

    def update(self, now, observation):
        """unknown打断静止证据，不能继承失联前的停稳；只允许moving延长初始等待。"""
        elapsed = now-self.started
        state = observation['state']
        if state == 'moving':
            self.soft_deadline = max(self.soft_deadline, now+6.)
            self.stable_since, self.stable_count = None, 0
        elif state == 'stationary' and elapsed >= self.grace:
            if self.stable_since is None:
                self.stable_since = now
            self.stable_count += 1
            if self.stable_count >= 3 and now-self.stable_since >= self.stable_seconds:
                return 'stalled'
        else:
            self.stable_since, self.stable_count = None, 0
        if elapsed >= self.total_limit:
            return 'movement_timeout'
        if now >= self.soft_deadline:
            return 'observation_unknown'
        return 'waiting'
