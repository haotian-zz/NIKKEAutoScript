"""原生点击反馈的真实图像几何与有界状态回归，不加载模型或输入驱动。"""

import unittest

import cv2
import numpy as np

from module.campaign.vision.native_motion import ClickProgress, NativeMotion


class NativeMotionTests(unittest.TestCase):
    """相机同移不能算角色移动，缺观测不能当停滞，移动续期仍有总时限。"""

    def field(self):
        """生成有纹理的多色背景，不依赖某种地貌的主色。"""
        random = np.random.default_rng(15)
        return cv2.GaussianBlur(random.integers(0, 255, (999, 1776, 3), dtype=np.uint8), (5, 5), 0)

    def test_camera_translation_does_not_count_as_squad_motion(self):
        """背景和箭头同时右移40px，扣镜头后应为静止。"""
        field = self.field()
        moved = cv2.warpAffine(field, np.float32([[1, 0, 40], [0, 1, 0]]), (1776, 999))
        result = NativeMotion(field, [880, 400]).observe(moved, [920, 400])
        self.assertEqual(result['state'], 'stationary')

    def test_squad_motion_with_camera_follow_is_detected(self):
        """镜头追随小队使箭头保持居中，背景左移仍代表真实走位。"""
        field = self.field()
        moved = cv2.warpAffine(field, np.float32([[1, 0, -40], [0, 1, 0]]), (1776, 999))
        result = NativeMotion(field, [880, 400]).observe(moved, [880, 400])
        self.assertEqual(result['state'], 'moving')
        self.assertGreater(result['movement_px'], 35)

    def test_missing_arrow_is_unknown(self):
        """箭头缺失不能以固定画面中心伪造静止或移动。"""
        field = self.field()
        self.assertEqual(NativeMotion(field).observe(field, [880, 400])['state'], 'unknown')

    def test_stall_requires_three_samples_and_time_span(self):
        """初始宽限后连续三帧且跨度达到两秒才允许设施诊断。"""
        progress = ClickProgress(0.)
        self.assertEqual(progress.update(1., dict(state='stationary')), 'waiting')
        self.assertEqual(progress.update(2., dict(state='stationary')), 'waiting')
        self.assertEqual(progress.update(3., dict(state='stationary')), 'waiting')
        self.assertEqual(progress.update(4., dict(state='stationary')), 'stalled')

    def test_unknown_breaks_stall_evidence(self):
        """失联前的静止样本不能与找回后的样本拼成停滞。"""
        progress = ClickProgress(0.)
        for now in [2., 3.]:
            progress.update(now, dict(state='stationary'))
        progress.update(4., dict(state='unknown'))
        self.assertEqual(progress.update(5., dict(state='stationary')), 'waiting')
        self.assertEqual(progress.update(6., dict(state='stationary')), 'waiting')

    def test_only_motion_extends_initial_wait_and_total_is_bounded(self):
        """真实移动可超过旧20秒；持续移动仍在60秒停止，未知不会无限等待。"""
        progress = ClickProgress(0.)
        for now in [18., 22., 26., 58.]:
            self.assertEqual(progress.update(now, dict(state='moving')), 'waiting')
        self.assertEqual(progress.update(60., dict(state='moving')), 'movement_timeout')
        self.assertEqual(ClickProgress(0.).update(20., dict(state='unknown')), 'observation_unknown')


if __name__ == '__main__':
    unittest.main()
