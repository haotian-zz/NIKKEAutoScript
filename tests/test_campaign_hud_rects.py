"""底部 HUD 只遮左下按钮和右下章节面板，画面底部中间的脚圈仍可点。"""
import unittest

from module.campaign.vision.enemy_ring import clickable_ring


class HudRectTests(unittest.TestCase):
    def test_bottom_middle_ring_is_clickable(self):
        """实录 2026-10-25（HARD 9）：敌人脚圈在底部中间 y≈905，整条底边遮罩时被拒、巡检看不到。"""
        self.assertTrue(clickable_ring([930., 875., 1020., 940.]))

    def test_bottom_corner_hud_still_masked(self):
        """左下返回按钮、右下章节面板上的框仍拒绝。"""
        self.assertFalse(clickable_ring([60., 905., 150., 960.]))
        self.assertFalse(clickable_ring([1560., 890., 1650., 950.]))
        self.assertFalse(clickable_ring([930., 940., 1020., 995.]))


if __name__ == '__main__':
    unittest.main()
