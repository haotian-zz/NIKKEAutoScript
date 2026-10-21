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
                    inliers=int(inside.sum()))


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
