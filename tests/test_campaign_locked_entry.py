"""前置主线未完成的备战弹窗：进入战斗被锁时仍认作备战弹窗，但只能关闭。"""

import unittest
from unittest.mock import patch

import numpy as np

from module.campaign.navigation import controls
from module.campaign.vision import battle


def frame(quick_orange, button=(196, 137, 64)):
    """1776×999 灰底画面，模板命中处（左上角）是按钮颜色；quick_orange 时把快速战斗按钮区域涂成实心橙色。"""
    image = np.full((999, 1776, 3), 90, np.uint8)
    height, width = controls.ENTER_BATTLE_TPL.shape[:2]
    image[:height, :width] = button
    if quick_orange:
        x1, y1, x2, y2 = battle.QUICK_BATTLE_ROI
        image[y1:y2, x1:x2] = (0, 128, 255)
    return image


class LockedEntryTests(unittest.TestCase):
    """模板分只决定候选区间，快速战斗按钮必须为橙色实心才放行。"""

    def score(self, image, template_value):
        with patch.object(controls.cv2, 'minMaxLoc', return_value=(0., template_value, (0, 0), (0, 0))):
            return controls.battle_popup_score(image)

    def test_locked_enter_with_quick_battle_is_accepted(self):
        """实录23-27：进入战斗被锁（模板0.757）、快速战斗橙色，按备战弹窗返回接受分并判为被锁。"""
        self.assertEqual(self.score(frame(True), .757), controls.LOCKED_ENTRY_ACCEPT)
        with patch.object(controls.cv2, 'minMaxLoc', return_value=(0., .757, (0, 0), (0, 0))):
            self.assertTrue(controls.entry_locked(frame(True)))

    def test_gray_locked_button_without_quick_battle_is_accepted(self):
        """实录 9-25 HARD：没打过的困难关没有快速战斗，进入战斗灰色压锁（模板0.761），同样判为被锁。"""
        self.assertEqual(self.score(frame(False, button=(75, 76, 76)), .761), controls.LOCKED_ENTRY_ACCEPT)
        with patch.object(controls.cv2, 'minMaxLoc', return_value=(0., .761, (0, 0), (0, 0))):
            self.assertTrue(controls.entry_locked(frame(False, button=(75, 76, 76))))

    def test_mid_score_without_quick_battle_is_not_a_popup(self):
        """同样的模板分但快速战斗不是橙色、按钮也不是灰色：保持原分数，不当作备战弹窗。"""
        self.assertAlmostEqual(self.score(frame(False), .757), .757)
        with patch.object(controls.cv2, 'minMaxLoc', return_value=(0., .757, (0, 0), (0, 0))):
            self.assertFalse(controls.entry_locked(frame(False)))

    def test_low_and_normal_scores_unchanged(self):
        """低于候选区间或正常可进入战斗时返回原模板分。"""
        self.assertAlmostEqual(self.score(frame(True), .5), .5)
        self.assertAlmostEqual(self.score(frame(True), .95), .95)

    def test_battle_state_never_quick_battles_locked_entry(self):
        """被锁弹窗判为 entry 但 quick=False：快速战斗会被游戏拒绝，接管时只能关闭。"""
        with patch.object(battle, 'battle_popup_score', return_value=controls.LOCKED_ENTRY_ACCEPT):
            state = battle.battle_state(frame(True))
        self.assertEqual(state['state'], 'entry')
        self.assertTrue(state['locked'])
        self.assertFalse(state['quick'])

if __name__ == '__main__':
    unittest.main()
