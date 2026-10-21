"""EX 补充视觉证据的文字、脚圈和小地图待处理保护。"""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from module.campaign.vision.ex_ring import minimap_enemy_hints, scene_ex_rings


class ExRingTests(unittest.TestCase):
    """几何候选不能独立发放敌人身份；未分类标记也不能作为零敌人略过。"""

    def field(self):
        """构造一个短标题及其下方的橙色脚圈，脚圈真值由绘图坐标独立指定。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        field[460:475, 580:606] = 255
        cv2.ellipse(field, (610, 550), (52, 22), 0, 0, 360, (100, 170, 255), 3)
        return field

    def test_explicit_ex_text_and_ellipse_are_both_required(self):
        """同时具备脚圈与高分 EX 文字才得到候选，落点接近独立绘制的真实圆心。"""
        reader = Mock()
        reader.paddleocr.recognize.return_value = [dict(rec_text='EX', rec_score=.99)]
        result = scene_ex_rings(self.field(), reader)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].center, (610, 550), atol=8.)
        for text, score in [('EXTRA', .99), ('EX', .94), ('EX', float('nan'))]:
            reader.paddleocr.recognize.return_value = [dict(rec_text=text, rec_score=score)]
            self.assertEqual(scene_ex_rings(self.field(), reader), [])

    def test_blank_field_does_not_run_ocr_or_invent_targets(self):
        """没有几何候选不启动 OCR，也不能凭孤立预期文字构造敌人。"""
        reader = Mock()
        self.assertEqual(scene_ex_rings(np.zeros((999, 1776, 3), np.uint8), reader), [])
        reader.paddleocr.recognize.assert_not_called()

    def test_separate_raw_letter_components_form_one_text_candidate(self):
        """未压缩的 E、X 中间有两像素空隙时仍能组合，不能依赖 JPEG 产生的连通伪影。"""
        field = self.field()
        field[460:475, 592:594] = 0
        reader = Mock()
        reader.paddleocr.recognize.return_value = [dict(rec_text='EX', rec_score=.99)]
        result = scene_ex_rings(field, reader)
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].center, (610, 550), atol=8.)

    def test_unclassified_minimap_triangle_remains_pending(self):
        """粉红敌人符号即使无模型类别仍保留待处理；固定资源图例不算敌人。"""
        panel = np.zeros((211, 218, 3), np.uint8)
        cv2.fillConvexPoly(panel, np.array([[60, 70], [80, 70], [70, 50]]), (220, 60, 220))
        cv2.rectangle(panel, (170, 190), (180, 200), (220, 60, 220), -1)
        hints = minimap_enemy_hints(panel)
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]['kind'], 'minimap_enemy_unclassified')
        self.assertLess(np.linalg.norm(np.asarray(hints[0]['position']) - [70, 63]), 3.)


if __name__ == '__main__':
    unittest.main()
