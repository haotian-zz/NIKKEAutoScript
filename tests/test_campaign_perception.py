"""小队箭头在旧遮罩外的定位回归；使用现有模板与真实精定位，不发送窗口输入。"""

import unittest
from unittest.mock import patch

import numpy as np

from module.campaign.vision import perception
from module.campaign.vision.detection import Detection


class SquadArrowTests(unittest.TestCase):
    """补查完整边缘箭头不能绕过 HUD 排除或唯一性规则。"""

    def setUp(self):
        """在右侧场景放置真实箭头模板，中心越过旧切片可完整检测的范围。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.candidate = self.draw(1610, 380)

    def draw(self, x, y):
        """在独立给定的位置绘制模板并返回模型框；精定位仍由生产实现完成。"""
        template = perception.arrow_template()
        height, width = template.shape
        self.field[y:y + height, x:x + width] = template[:, :, None]
        return Detection('scene_squad_arrow', .99, (x, y, x + width, y + height))

    def test_complete_arrow_beyond_legacy_mask_can_be_refined(self):
        """旧遮罩漏掉的完整箭头经整图检测和模板复核恢复精确中心。"""
        with patch.object(perception, 'detect_scene', return_value=[]), \
                patch.object(perception, 'detect_grid', return_value=[self.candidate]):
            np.testing.assert_allclose(perception.squad_arrow(self.field), [1631., 398.5])

    def test_hud_and_ambiguous_arrows_are_rejected(self):
        """HUD 上的假箭头或两个可信场景箭头均不得给出小队坐标。"""
        hud = self.draw(1550, 100)
        second = self.draw(800, 500)
        for candidates in ([hud], [self.candidate, second]):
            with patch.object(perception, 'detect_scene', return_value=[]), \
                    patch.object(perception, 'detect_grid', return_value=candidates):
                self.assertIsNone(perception.squad_arrow(self.field))

    def test_verified_legacy_arrow_does_not_add_grid_inference(self):
        """已经定位的小队沿用原路径，不在每帧重复整图推理。"""
        with patch.object(perception, 'detect_scene', return_value=[self.candidate]), \
                patch.object(perception, 'detect_grid') as detect:
            np.testing.assert_allclose(perception.squad_arrow(self.field), [1631., 398.5])
        detect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
