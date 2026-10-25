"""走不到／点了没反应的位置按场景配准跟到新画面；不加载模型或连接游戏。"""
import unittest

import cv2
import numpy as np

from module.campaign.rejected_targets import REJECT_RADIUS, RejectedTargets, static_view


def textured(shift=(0, 0)):
    """带纹理的 1776×999 画面；shift 模拟镜头平移后的同一场景。"""
    rng = np.random.default_rng(7)
    big = cv2.GaussianBlur(rng.integers(0, 255, (1199, 1976, 3), dtype=np.uint8), (5, 5), 0)
    x, y = 100 - shift[0], 100 - shift[1]
    return np.ascontiguousarray(big[y:y+999, x:x+1776])


class RejectedTargetsTests(unittest.TestCase):
    def test_position_follows_camera_pan(self):
        """镜头平移后记下的位置跟着走，附近的新编号被跳过，远处的保留。"""
        rejected = RejectedTargets()
        rejected.add(textured(), [900., 500.], 'stalled')
        moved = textured((60, -40))
        (point, reason), = rejected.points(moved)
        np.testing.assert_allclose(point, [960., 460.], atol=3.)
        near = dict(id='ring-40-0', click=[955., 470.])
        far = dict(id='ring-40-1', click=[960.+REJECT_RADIUS*2, 460.])
        kept, skipped = rejected.filter(moved, [near, far])
        self.assertEqual(kept, [far])
        self.assertEqual((skipped[0][0], skipped[0][2]), (near, 'stalled'))

    def test_moved_record_marks_without_skipping(self):
        """走过才停下的位置只作标记：不跳过候选，但能查到附近已有标记（第二次停下才记走不到）。"""
        rejected = RejectedTargets()
        rejected.add(textured(), [900., 500.], 'moved')
        moved = textured((60, -40))
        target = dict(id='ring-40-0', click=[955., 470.])
        self.assertEqual(rejected.filter(moved, [target]), ([target], []))
        self.assertTrue(rejected.near(moved, target['click'], ('moved',)))
        self.assertFalse(rejected.near(moved, [960.+REJECT_RADIUS*2, 460.], ('moved',)))
        self.assertFalse(rejected.near(moved, target['click'], ('stalled',)))

    def test_unregistered_record_excludes_nothing(self):
        """配准不上（换了场景）的记录不排除任何目标。"""
        rejected = RejectedTargets()
        rejected.add(textured(), [900., 500.], 'no_effect')
        other = cv2.GaussianBlur(np.random.default_rng(1).integers(0, 255, (999, 1776, 3), dtype=np.uint8), (5, 5), 0)
        target = dict(id='ring-1', click=[900., 500.])
        self.assertEqual(rejected.filter(other, [target]), ([target], []))

    def test_empty_records_skip_registration(self):
        rejected = RejectedTargets()
        target = dict(id='ring-1', click=[900., 500.])
        self.assertEqual(rejected.filter(np.zeros((999, 1776, 3), np.uint8), [target]), ([target], []))

    def test_static_view(self):
        """镜头没动才算静止；平移 100px 不算。"""
        self.assertTrue(static_view(textured(), textured((3, 2))))
        self.assertFalse(static_view(textured(), textured((100, 0))))
        self.assertFalse(static_view(np.zeros((999, 1776, 3), np.uint8), textured()))


if __name__ == '__main__':
    unittest.main()
