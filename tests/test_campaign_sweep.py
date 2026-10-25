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

    def test_parallax_falls_back_to_dominant_shift_but_blocks_clearance(self):
        """实录22-36：严格配准缺背景内点时按主平移继续找敌人，覆盖记存疑；主平移也测不到才停止。"""
        failed = SweepUnconfirmed('镜头运动缺少分布充分的背景内点，未确认巡检覆盖。')
        shift = Mock(return_value=np.array([-470., 4.]))
        with patch.object(sweep_module, 'camera_motion', side_effect=failed), \
                patch.dict('sys.modules', {'module.campaign.switch_search': SimpleNamespace(content_shift=shift)}):
            np.testing.assert_allclose(self.sweep.measure_motion('a', 'b', [480., 0.]), [-470., 4.])
            self.assertEqual(len(self.sweep.coverage_warnings), 1)
            shift.return_value = None
            with self.assertRaisesRegex(SweepUnconfirmed, '背景内点'):
                self.sweep.measure_motion('a', 'b', [480., 0.])
        with self.assertRaisesRegex(SweepUnconfirmed, '斜向'):
            self.sweep.run()

    def test_fog_edge_pans_back_and_continues_rows(self):
        """实录 2026-10-25（HARD 9）：拖进左侧雾里测不到位移时拖回一格当作边界，继续往下扫，不整轮中止。"""
        from module.campaign.sweep import SweepMotionLost
        real = self.sweep.move.side_effect

        def move(drag, before=None):
            """往左拖到镜头 x=0 再拖就进雾：测不到位移。"""
            if drag[0] > 0 and self.position[0] <= 0.:
                self.sweep.steps += 1
                raise SweepMotionLost('镜头运动缺少分布充分的背景内点，未确认巡检覆盖。')
            return real(drag, before)

        self.sweep.move = Mock(side_effect=move)
        self.sweep.pan = Mock()
        self.sweep.pan_anchor = Mock(return_value=None)
        self.runner.map_frame.return_value = np.zeros((999, 1776, 3), np.uint8)
        with patch.object(sweep_module.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaisesRegex(SweepUnconfirmed, '雾边'):
                self.sweep.run()
        self.assertGreaterEqual(self.sweep.proof['rows'], 3)
        for point in ((0., 520.), (960., 520.)):
            self.assertIn(point, self.visits)
        np.testing.assert_allclose(self.sweep.pan.call_args_list[0][0][0], [-480., 0.])
        self.assertTrue(self.sweep.coverage_warnings)

    def test_fog_edge_limit_stops(self):
        """一直测不到位移不是地图边缘，超过次数仍停止。"""
        from module.campaign.sweep import SweepMotionLost
        self.sweep.move = Mock(side_effect=SweepMotionLost('背景内点'))
        self.sweep.pan = Mock()
        self.sweep.pan_anchor = Mock(return_value=None)
        self.sweep.fog_edges = sweep_module.FOG_EDGE_LIMIT
        self.runner.map_frame.return_value = np.zeros((999, 1776, 3), np.uint8)
        with patch.object(sweep_module.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaisesRegex(SweepUnconfirmed, '雾边测不到'):
                self.sweep.run()
        self.sweep.pan.assert_not_called()

    def test_unseen_squad_tracked_by_accumulated_content_shift(self):
        """实录22-36：拖几屏后箭头被任务栏旁的 HUD 排除；按看到时的位置加累计位移推算，画面内避开，画面外判离屏。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        arrow = Mock(return_value=np.array([600., 400.]))
        with patch.object(sweep_module, 'squad_arrow', arrow), \
                patch.object(sweep_module, 'squad_offscreen', return_value=None), \
                patch.object(sweep_module.goto, 'ARROW_GROUND_OFFSET', 90.):
            np.testing.assert_allclose(self.sweep.pan_anchor(field), [600., 490.])
            arrow.return_value = None
            self.sweep.content = np.array([1060., -90.])
            np.testing.assert_allclose(self.sweep.pan_anchor(field), [1660., 400.])
            self.sweep.content = np.array([1400., -90.])
            self.assertIsNone(self.sweep.pan_anchor(field))
            self.sweep.squad_seen, self.sweep.occlusion_checked = None, True
            with self.assertRaises(SweepUnconfirmed):
                self.sweep.pan_anchor(field)

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
