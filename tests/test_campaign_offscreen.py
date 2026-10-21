"""固定UI遮挡与场景缺口必须区分，遮住尾巴时不能推测小队离屏。"""

import unittest

import numpy as np

from module.campaign.vision import offscreen


class OffscreenOcclusionTests(unittest.TestCase):
    """直接验证圆环原始像素测量，避免用降低置信度解释漏检。"""

    def marker(self):
        """构造细环与朝左尾巴，保持标记像素独立于待测遮挡掩码。"""
        yy, xx = np.mgrid[:220, :220]
        distance = np.hypot(xx-110, yy-110)
        angle = np.degrees(np.arctan2(yy-110, xx-110)) % 360
        ring = np.abs(distance-offscreen.RADIUS) <= 1.
        tail = ((distance >= offscreen.RADIUS+2) & (distance <= offscreen.RADIUS+10)
                & (angle >= 167) & (angle <= 193))
        return ring | tail

    def test_known_top_occlusion_uses_visible_arc(self):
        """小地图遮住上方少量圆弧可恢复，未登记的同样缺口仍拒绝。"""
        mask = self.marker()
        visible = np.ones(mask.shape, bool)
        visible[:90] = False
        mask[~visible] = False
        self.assertFalse(offscreen.accepted(offscreen.measure(mask, 110, 110), 80, 400))
        self.assertTrue(offscreen.accepted(offscreen.measure(mask, 110, 110, visible), 80, 400))

    def test_hidden_tail_and_mostly_hidden_ring_are_rejected(self):
        """朝向证据或大部分圆环不可见时不允许靠其余亮弧放行。"""
        mask = self.marker()
        for visible in (np.indices(mask.shape)[1] >= 74, np.indices(mask.shape)[0] >= 110):
            self.assertFalse(offscreen.accepted(offscreen.measure(mask & visible, 110, 110, visible), 80, 400))

    def test_bright_facility_and_ring_without_tail_stay_rejected(self):
        """亮圆形设施与无尾巴的闭环均不是小队离屏标记。"""
        yy, xx = np.mgrid[:220, :220]
        for mask in (np.hypot(xx-110, yy-110) < 35,
                     np.abs(np.hypot(xx-110, yy-110)-offscreen.RADIUS) <= 1.):
            self.assertFalse(offscreen.accepted(offscreen.measure(mask, 110, 110), 80, 400))
