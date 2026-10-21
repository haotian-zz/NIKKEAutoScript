"""全章清怪巡检的覆盖、遇敌与失败边界回归，不发送真实输入。"""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign import sweep as sweep_module
    from module.campaign.sweep import ChapterSweep, SweepUnconfirmed, camera_motion


class CameraMotionTests(unittest.TestCase):
    """运动证据不足时不得将一个静止或跳变画面当成全章边界。"""

    def registration(self, delta):
        """建立有画面内背景支持的精确平移真值。"""
        matrix = np.eye(3)
        matrix[:2, 2] = delta
        hull = np.float32([[100., 200.], [1600., 200.], [1600., 850.], [100., 850.]]).reshape(-1, 1, 2)
        return matrix, hull, hull + np.float32(delta)

    def test_correct_motion_and_static_are_distinct(self):
        """同一帧可作边界候选；正常平移必须保留真实非零响应。"""
        for delta in ([0., 0.], [300., 0.], [0., -240.]):
            with patch.object(sweep_module, 'scene_registration', return_value=self.registration(delta)):
                np.testing.assert_allclose(camera_motion(None, None, delta if any(delta) else [480., 0.]), delta)

    def test_projective_camera_response_is_not_forced_into_translation(self):
        """真实相机的剪切或透视运动按完整脚印重叠验证，不因上下位移不同就误停。"""
        matrix = np.array([[1., .17, 150.], [0., 1., 0.], [0., 0., 1.]])
        hull = np.float32([[0., 100.], [1700., 100.], [1700., 900.], [0., 900.]]).reshape(-1, 1, 2)
        target_hull = cv2.perspectiveTransform(hull, matrix)
        with patch.object(sweep_module, 'scene_registration', return_value=(matrix, hull, target_hull)):
            delta = camera_motion(None, None, [480., 0.])
        self.assertGreater(delta[0], 200.)
        self.assertAlmostEqual(delta[1], 0.)

    def test_bad_direction_jump_and_missing_support_are_rejected(self):
        """反向运动、横向漂移、大幅跳图和内点支持过窄均不能连接覆盖。"""
        for delta in ([-200., 0.], [400., 120.], [700., 0.]):
            with patch.object(sweep_module, 'scene_registration', return_value=self.registration(delta)):
                with self.assertRaises(SweepUnconfirmed):
                    camera_motion(None, None, [480., 0.])
        matrix, _, _ = self.registration([100., 0.])
        hull = np.float32([[0., 0.], [100., 0.], [100., 100.], [0., 100.]]).reshape(-1, 1, 2)
        with patch.object(sweep_module, 'scene_registration', return_value=(matrix, hull, hull)):
            with self.assertRaises(SweepUnconfirmed):
                camera_motion(None, None, [480., 0.])

    def test_terrain_on_one_side_keeps_two_dimensional_support(self):
        """地图边缘另一侧为空白时，真实地形的二维内点仍能验证运动，细线支持则拒绝。"""
        matrix = np.eye(3)
        matrix[0, 2] = 150.
        for height, accepted in ((700., True), (30., False)):
            hull = np.float32([[950., 150.], [1650., 150.],
                               [1650., 150. + height], [950., 150. + height]]).reshape(-1, 1, 2)
            destination = cv2.perspectiveTransform(hull, matrix)
            with patch.object(sweep_module, 'scene_registration', return_value=(matrix, hull, destination)):
                if accepted:
                    np.testing.assert_allclose(camera_motion(None, None, [480., 0.]), [150., 0.])
                else:
                    with self.assertRaises(SweepUnconfirmed):
                        camera_motion(None, None, [480., 0.])

    def test_bounded_diagonal_edge_response_is_measured_for_exploration(self):
        """相机边缘的实际斜向夹持可继续探索；该位移不能单独作为规整行扫描证据。"""
        with patch.object(sweep_module, 'scene_registration', return_value=self.registration([17., 31.])):
            np.testing.assert_allclose(camera_motion(None, None, [480., 0.]), [17., 31.])

    def test_oblique_collection_does_not_relax_clearance_or_overlap(self):
        """设施采集可观测斜向边缘，但默认清怪仍拒绝；反向和跳图仍禁止。"""
        with patch.object(sweep_module, 'scene_registration', return_value=self.registration([-77., -89.])):
            with self.assertRaises(SweepUnconfirmed):
                camera_motion(None, None, [-480., 0.])
            np.testing.assert_allclose(camera_motion(None, None, [-480., 0.], allow_oblique=True), [-77., -89.])
        for delta in ([200., 0.], [-700., 100.]):
            with patch.object(sweep_module, 'scene_registration', return_value=self.registration(delta)):
                with self.assertRaises(SweepUnconfirmed):
                    camera_motion(None, None, [-480., 0.], allow_oblique=True)


class SceneRegistrationTests(unittest.TestCase):
    """独立合成纹理和已知真实平移验证生产背景配准及红圈跟踪。"""

    def test_known_camera_translation_tracks_rings(self):
        """几何真值来自合成图变换，检测特征及关联使用真实 SIFT 和生产实现。"""
        from module.campaign.vision.enemy_ring import RingFrame, scene_registration, scene_tracks
        rng = np.random.default_rng(371)
        image = cv2.GaussianBlur(rng.integers(0, 256, (900, 1600), dtype=np.uint8), (5, 5), 1.)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 35], [0, 1, -20]]), (1600, 900))
        frames = []
        for field, centers in ((image, [[700, 500], [1050, 650]]),
                               (shifted, [[735, 480], [1085, 630]])):
            keys, descriptors = cv2.SIFT_create(nfeatures=1800).detectAndCompute(field, None)
            frames.append(RingFrame(np.asarray(centers, float), np.asarray([key.pt for key in keys]),
                                    descriptors, np.zeros(2), [], []))
        matrix, _, _ = scene_registration(*frames)
        expected = np.float32([[[400, 400], [800, 500], [1200, 700]]])
        actual = cv2.perspectiveTransform(expected, matrix)
        np.testing.assert_allclose(actual, expected + [35, -20], atol=.5)
        old, tracked = scene_tracks(*frames)
        self.assertTrue(tracked.all())
        np.testing.assert_allclose(old, frames[0].centers, atol=.5)


class ChapterSweepTests(unittest.TestCase):
    """只有逐行覆盖和当前零敌人证据同时满足才确认清空；任何故障保持未完成。"""

    def setUp(self):
        """使用二维有界相机替身验证蛇形遍历，边界和移动仍由生产状态机判断。"""
        self.runner = Mock()
        self.runner.session.evidence.folder = Path('/unused/mock-evidence')
        self.sweep = ChapterSweep(self.runner)
        self.position = np.array([480., 260.])
        self.visits = []

        def move(drag, before=None):
            """模拟宽 960、高 520 的相机可移动区，移动夹在实测边界内。"""
            old = self.position.copy()
            self.position = np.clip(old - np.asarray(drag), [0., 0.], [960., 520.])
            delta = old - self.position
            self.sweep.steps += 1
            if np.linalg.norm(delta) >= 3.:
                self.sweep.axes.add(int(np.argmax(np.abs(drag))))
            self.visits.append(tuple(self.position))
            return delta

        self.sweep.move = Mock(side_effect=move)

    def test_missing_minimap_is_not_zero_enemies(self):
        """面板消失时明确未完成，不调用模型把空裁图解释成敌人数量零。"""
        self.runner.map_frame.return_value = np.zeros((999, 1776, 3), np.uint8)
        self.runner.deferred = set()
        self.runner.memory.observe.return_value = []
        with patch.object(sweep_module, 'ring_frame', return_value=Mock()), \
                patch.object(sweep_module.goto, 'battle_popup_score', return_value=0.), \
                patch.object(sweep_module, 'detect_minimap') as detect:
            with self.assertRaisesRegex(SweepUnconfirmed, '小地图未确认'):
                self.sweep.sample()
        detect.assert_not_called()

    def test_serpentine_visits_all_rows_before_verified_empty(self):
        """左上、右上及下方各行均实际到达，最后一行到边界后才允许清空。"""
        result = self.sweep.run()
        self.assertEqual(result['state'], 'verified_empty')
        self.assertEqual(result['proof']['rows'], 3)
        for point in ((0., 0.), (960., 0.), (0., 260.), (960., 260.), (0., 520.), (960., 520.)):
            self.assertIn(point, self.visits)
        self.runner.session.record_evidence.assert_called()

    def test_any_remaining_normal_or_ex_marker_blocks_clearance(self):
        """即使所有镜头边界确认，任意普通或 EX 待处理标记仍阻止换章。"""
        self.sweep.hints = [dict(view=1, minimap=[dict(kind='minimap_enemy_ex', position=[20, 30])])]
        with self.assertRaisesRegex(SweepUnconfirmed, '仍发现'):
            self.sweep.run()
        self.assertEqual(self.sweep.proof['state'], 'incomplete')
        self.assertTrue(self.sweep.proof['pending'])

    def test_static_views_without_axis_response_are_not_complete(self):
        """驱动未实际移动、或只有一轴响应时，重复静止不能伪装成四边完整覆盖。"""
        self.sweep.move = Mock(return_value=np.zeros(2))
        with self.assertRaisesRegex(SweepUnconfirmed, '两轴'):
            self.sweep.run()
        self.assertEqual(self.sweep.proof['state'], 'incomplete')

    def test_nonaxial_response_never_proves_full_clearance(self):
        """即使走完模拟四边且无敌人，有斜向覆盖存疑也只能报告未完成。"""
        self.sweep.coverage_warnings = [dict(pan=1, drag=[480., 0.], actual=[17., 31.])]
        with self.assertRaisesRegex(SweepUnconfirmed, '斜向'):
            self.sweep.run()
        self.assertEqual(self.sweep.proof['state'], 'incomplete')
        self.assertEqual(self.sweep.proof['unconfirmed_motion'], self.sweep.coverage_warnings)

    def test_new_enemy_interrupts_survey_for_immediate_processing(self):
        """发现当前可点击敌人立即结束巡检，战斗之后由新巡检重新核验全章。"""
        def found(drag, before=None):
            """模拟第一步镜头移动就发现新敌人。"""
            self.sweep.found = dict(state='target', targets=[dict(id='new', click=[700., 500.])])
            return None
        self.sweep.move = Mock(side_effect=found)
        self.assertEqual(self.sweep.run()['state'], 'target')
        self.assertEqual(self.sweep.move.call_count, 1)
        self.assertEqual(self.sweep.proof['state'], 'incomplete')

    def test_input_failure_is_not_treated_as_an_edge(self):
        """手势失败向上传播，已有待处理提示仍存入证据，不能得到空图结论。"""
        self.sweep.hints = [dict(view=0, minimap=[dict(kind='minimap_enemy_ex')])]
        self.sweep.move = Mock(side_effect=RuntimeError('mouse_swipe failed'))
        with self.assertRaisesRegex(RuntimeError, 'mouse_swipe'):
            self.sweep.run()
        self.assertEqual(self.sweep.proof['state'], 'incomplete')
        self.assertEqual(self.sweep.proof['pending'], self.sweep.hints)


if __name__ == '__main__':
    unittest.main()
