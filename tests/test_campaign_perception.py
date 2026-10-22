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



class MinimapSquadGroundTests(unittest.TestCase):
    """箭头被遮挡时由紧凑小地图圆环推算小队脚下位置；面板不可见或圆环不唯一不给锚点。"""

    def setUp(self):
        self.field = np.zeros((999, 1776, 3), np.uint8)
        patcher = patch.multiple('module.campaign.vision.minimap', minimap_control_visible=lambda crop: True,
                                 map_visible=lambda crop: True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ring_maps_to_ground_and_arrow_wins(self):
        """实录22-36：小队在半透明升降台里，圆环 (106.9,106.0) 推算到画面中央附近；有箭头时仍用箭头。"""
        with patch.object(perception, 'detect_markers', return_value=([np.array([106.9, 106.])], [])), \
                patch.object(perception, 'squad_arrow', return_value=None):
            ground, source = perception.squad_ground(self.field)
        self.assertEqual(source, 'minimap')
        np.testing.assert_allclose(ground, [885., 517.], atol=2.)
        with patch.object(perception, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(perception, 'detect_markers') as markers:
            ground, source = perception.squad_ground(self.field)
        self.assertEqual(source, 'arrow')
        markers.assert_not_called()

    def test_missing_or_ambiguous_ring_gives_no_anchor(self):
        """没有圆环或多个圆环都不推算。"""
        for players in ([], [np.array([100., 100.]), np.array([120., 90.])]):
            with patch.object(perception, 'detect_markers', return_value=(players, [])):
                self.assertIsNone(perception.minimap_squad_ground(self.field))

if __name__ == '__main__':
    unittest.main()
