"""地图配准的投影边界回归，不读取账号截图或发送游戏输入。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from module.campaign.vision import matching
from module.campaign.vision.matching import template_matrix


class ProjectionBoundaryTests(unittest.TestCase):
    """有限角点不能掩盖面板内部的投影奇点。"""

    def test_internal_horizon_is_rejected_even_when_corners_are_finite(self):
        """模拟展开面板伪高分的来源：四角有限，但深度在面板内变号。"""
        projection = np.array([[1., 0, 0], [0, 1., 0], [0, .01, -1.]])
        self.assertIsNone(template_matrix(projection, 1., 1., (462, 486)))

    def test_horizon_on_panel_edge_is_rejected(self):
        """深度恰为零的角点也不能交给 OpenCV 变成有限伪坐标。"""
        projection = np.array([[1., 0, 0], [0, 1., 0], [0, .01, 0.]])
        self.assertIsNone(template_matrix(projection, 1., 1., (462, 486)))

    def test_equivalent_negative_homography_remains_valid(self):
        """整体乘负号不改变几何，不能误拒绝合法单应。"""
        positive = template_matrix(np.eye(3), 1., 1., (462, 486))
        negative = template_matrix(-np.eye(3), 1., 1., (462, 486))
        self.assertIsNotNone(positive)
        self.assertEqual(positive[1], negative[1])
        points = np.array([[[30., 40], [210., 350]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(points, positive[0]),
                                   cv2.perspectiveTransform(points, negative[0]))

    def test_nonfinite_projection_is_rejected(self):
        """损坏的投影不能变成尺寸溢出或参与搜索。"""
        projection = np.eye(3)
        projection[0, 0] = np.nan
        self.assertIsNone(template_matrix(projection, 1., 1., (462, 486)))

    def test_failed_compact_snapshot_is_not_computed_twice(self):
        """初始拒绝后必须换成新展开视图，不能再对原视图做全网格搜索。"""
        source = ast.parse((Path(__file__).resolve().parents[1] / 'module/campaign/campaign.py').read_text('utf-8'))
        node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef)
                    and n.name == 'confirm_squad_location')
        namespace = dict(logger=Mock(), points_text=str, MINIMAP_MODES={}, LOCALIZATION={},
                         label=lambda table, value: value, SQUAD_LOCATE_ATTEMPTS=4,
                         MINIMAP_PAN_DIRECTIONS=((1, 0),), publish_preview_frame=Mock(), cv2=cv2, time=time)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'confirm_squad_location', 'exec'), namespace)
        accepted = dict(localization='accepted', squad=[20, 40], minimap={'mode': 'expanded'})
        runner = SimpleNamespace(current_chapter=20, current_difficulty='normal',
                                 expand_minimap=Mock(return_value=True), collapse_minimap=Mock(),
                                 pan_expanded_minimap=Mock(), campaign_source=Mock(),
                                 observer=SimpleNamespace(observe=Mock(return_value=accepted)),
                                 device=SimpleNamespace(screenshot=Mock(), image=np.zeros((8, 8, 3), np.uint8)))
        self.assertIs(namespace['confirm_squad_location'](runner, {'localization': 'needs_review'}), accepted)
        runner.expand_minimap.assert_called_once()
        runner.observer.observe.assert_called_once()
        runner.device.screenshot.assert_called_once()
        runner.pan_expanded_minimap.assert_not_called()
        runner.collapse_minimap.assert_called_once()


class ParallelSearchTests(unittest.TestCase):
    """使用同一合成底图比较并行与串行的完整定位结果。"""

    def test_parallel_search_keeps_candidate_scores_and_rejection(self):
        """并行只改变调度；包括歧义拒绝在内的判定不能变化。"""
        terrain = np.zeros((300, 360), np.float32)
        cv2.line(terrain, (35, 75), (330, 170), 1., 19)
        cv2.line(terrain, (150, 20), (190, 280), 1., 25)
        cv2.circle(terrain, (255, 220), 35, 1., 12)
        road = (terrain[70:170, 100:200] * 255).astype(np.uint8)
        valid = np.full(road.shape, 255, np.uint8)
        matcher = matching.MapMatcher.__new__(matching.MapMatcher)
        matcher.projection = np.eye(3)
        matcher.terrain = terrain
        matcher.target = cv2.copyMakeBorder(cv2.resize(terrain, None, fx=.25, fy=.25,
                                                       interpolation=cv2.INTER_AREA),
                                           matching.PAD, matching.PAD, matching.PAD, matching.PAD,
                                           cv2.BORDER_CONSTANT, value=0)
        matcher.pad = 900
        matcher.large = cv2.copyMakeBorder(terrain, 900, 900, 900, 900, cv2.BORDER_CONSTANT, value=0)
        matcher.digest = 'synthetic'
        grid = [(a, b) for a in (.8, 1., 1.2) for b in (.8, 1., 1.2)]
        results = []
        for workers in (1, 4):
            matcher.frame_cache = {}
            with patch.object(matching, 'PARAMETER_GRID', grid), patch.object(matching, 'SEARCH_WORKERS', workers):
                results.append(matcher.match_panel(None, road, valid, np.array([50., 50.])))
        self.assertEqual(results[0], results[1])


class RectifiedTrackingTests(unittest.TestCase):
    """已知平面运动的合成真值检验，覆盖透视校正、双有效域和错误运动先验。"""

    def pair(self):
        """以已知平面位移渲染两张透视地图，返回第一帧投影与第二帧真值。"""
        rng = np.random.default_rng(22)
        world = np.zeros((600, 600), np.uint8)
        for _ in range(18):
            cv2.line(world, tuple(rng.integers(40, 560, 2)), tuple(rng.integers(40, 560, 2)), 255, 8)
        cv2.polylines(world, [np.array([[60, 50], [140, 50], [140, 140], [180, 140]])], False, 255, 8)
        cv2.line(world, (70, 90), (160, 170), 255, 8)
        projection = np.array([[1., .1, 25.], [0., 1.6, 25.], [0., .003, 1.]])
        moved = np.array([[1., 0., 35.], [0., 1., -30.], [0., 0., 1.]]) @ projection
        first, second = [cv2.warpPerspective(world, np.linalg.inv(matrix), (218, 211),
                                             flags=cv2.INTER_NEAREST) for matrix in (projection, moved)]
        valid = np.full(first.shape, 255, np.uint8)
        valid[:4] = valid[-4:] = valid[:, :4] = valid[:, -4:] = 0
        return first, second, valid, projection, moved

    def test_recovers_known_planar_motion_after_rectification(self):
        """前后帧不是原图纯平移，校正后的矩阵必须回到生成真值而非只报高 IoU。"""
        first, second, valid, projection, moved = self.pair()
        result = matching.register_rectified(first, second, valid, valid, projection, [-35., 30.])
        self.assertIsNotNone(result)
        points = np.array([[[45., 50.], [110., 110.], [175., 170.]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(points, result['matrix']),
                                   cv2.perspectiveTransform(points, moved), atol=2.)
        self.assertGreater(result['iou'], .9)
        self.assertLessEqual(result['round_trip_error'], 2.)

    def test_motion_disagreeing_with_gesture_is_rejected(self):
        """道路拟合即使有解，也不能用与实际手势明显相反的位移继续链式跟踪。"""
        first, second, valid, projection, _ = self.pair()
        self.assertIsNone(matching.register_rectified(first, second, valid, valid, projection, [90., -80.]))

    def test_each_frame_uses_its_own_valid_mask(self):
        """后一帧被遮挡的道路不应作为当前帧背景计入交并比。"""
        first = np.zeros((8, 8), np.uint8)
        first[2, 2] = 255
        second = first.copy()
        second[5, 5] = 255
        valid = np.full(first.shape, 255, np.uint8)
        second_valid = valid.copy()
        second_valid[5, 5] = 0
        self.assertEqual(matching.overlap(first, second, valid, (0, 0), second_valid), (1., 1))

    def test_long_straight_road_is_not_unique_motion_evidence(self):
        """沿直路方向不可观察时拒绝链式跟踪，不能靠高重合度猜测沿向位移。"""
        first = np.zeros((180, 180), np.uint8)
        cv2.line(first, (15, 90), (165, 90), 255, 5)
        valid = np.full(first.shape, 255, np.uint8)
        self.assertIsNone(matching.register_rectified(first, first, valid, valid, np.eye(3), [15., 0.]))


class ConfidenceCacheTests(unittest.TestCase):
    """画面变化后的全局歧义必须重新判断，缓存只服务完全相同的观测。"""

    def setUp(self):
        """建立独特道路和两处相同道路，副本间隔超过独立竞争位置的排除半径。"""
        rng = np.random.default_rng(40)
        self.terrain = np.zeros((520, 800), np.float32)
        self.roads = []
        for _ in range(2):
            road = np.zeros((100, 100), np.uint8)
            for _ in range(12):
                cv2.line(road, tuple(rng.integers(8, 92, 2)), tuple(rng.integers(8, 92, 2)), 255, 6)
            self.roads.append(road)
        for road, (x, y) in ((self.roads[0], (40, 40)), (self.roads[1], (240, 300)),
                             (self.roads[1], (540, 300))):
            self.terrain[y:y + 100, x:x + 100] = road / 255
        self.valid = np.full((100, 100), 255, np.uint8)
        self.player = np.array([50., 50.])
        patcher = patch.object(matching, 'PARAMETER_GRID', [(1., 1.)])
        patcher.start()
        self.addCleanup(patcher.stop)

    def matcher(self):
        """只绕过地图包加载，搜索、评分、置信度和缓存均执行正式实现。"""
        result = matching.MapMatcher.__new__(matching.MapMatcher)
        result.terrain, result.projection, result.pad = self.terrain, np.eye(3), 900
        result.large = cv2.copyMakeBorder(self.terrain, 900, 900, 900, 900, cv2.BORDER_CONSTANT, value=0)
        result.target = cv2.copyMakeBorder(cv2.resize(self.terrain, None, fx=.25, fy=.25),
                                          matching.PAD, matching.PAD, matching.PAD, matching.PAD,
                                          cv2.BORDER_CONSTANT, value=0)
        result.digest = 'synthetic-confidence'
        return result

    def test_changed_road_cannot_inherit_unique_confidence(self):
        """独特图之后的重复道路必须与冷启动一样拒绝，即使尺度和最佳 IoU 完全一致。"""
        matcher = self.matcher()
        self.assertEqual(matcher.match_panel(None, self.roads[0], self.valid, self.player)['status'], 'accepted')
        warm = matcher.match_panel(None, self.roads[1], self.valid, self.player)
        cold = self.matcher().match_panel(None, self.roads[1], self.valid, self.player)
        self.assertEqual(warm, cold)
        self.assertEqual(warm['status'], 'needs_review')
        self.assertAlmostEqual(warm['coarse_margin'], 0., delta=1e-6)

    def test_cache_requires_identical_masks_and_anchors_and_returns_copy(self):
        """缓存不受调用方修改影响，有效域、实测圆环和位置约束变化均重新计算。"""
        matcher = self.matcher()
        with patch.object(matching, 'iou_surface', wraps=matching.iou_surface) as search:
            result = matcher.match_panel(None, self.roads[0], self.valid, self.player)
            count = search.call_count
            result['position'][0] = -999
            cached = matcher.match_panel(None, self.roads[0], self.valid.copy(), self.player.copy())
            self.assertEqual(search.call_count, count)
            self.assertGreater(cached['position'][0], 0)
            changed = self.valid.copy()
            changed[:5] = 0
            matcher.match_panel(None, self.roads[0], changed, self.player)
            self.assertGreater(search.call_count, count)
            count = search.call_count
            matcher.match_panel(None, self.roads[0], changed, self.player + [1, 0])
            self.assertGreater(search.call_count, count)
            count = search.call_count
            matcher.match_panel(None, self.roads[0], self.valid, self.player, expected_position=[90, 90])
            self.assertGreater(search.call_count, count)


if __name__ == '__main__':
    unittest.main()
