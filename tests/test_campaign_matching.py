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
            with patch.object(matching, 'PARAMETER_GRID', grid), patch.object(matching, 'SEARCH_WORKERS', workers):
                results.append(matcher.match_panel(None, road, valid, np.array([50., 50.])))
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
