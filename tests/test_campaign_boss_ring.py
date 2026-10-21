"""大型入口补充语义的关键反例：骷髅和实际脚圈必须在同一新帧同时成立。"""

from pathlib import Path
import unittest

import cv2
import numpy as np

from module.campaign.vision.boss_ring import boss_ring_candidates


class BossRingTests(unittest.TestCase):
    """区分图标、地面圆环与真实联合候选，不加载模型或发送游戏输入。"""

    def scene(self, skull=True, ring=True):
        """在固定UI外放置真实二值图标与合成发光环，位置不对应任何已知章节。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        path = Path(__file__).resolve().parents[1] / 'module/campaign/vision/assets/boss_skull_mask.png'
        glyph = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if skull:
            height, width = glyph.shape
            field[350:350+height, 800:800+width] = cv2.cvtColor(glyph, cv2.COLOR_GRAY2BGR)
        if ring:
            cv2.ellipse(field, (821, 455), (74, 44), 0, 0, 360, (0, 90, 255), 5)
        return field

    def test_two_independent_visual_parts_are_required(self):
        """孤立骷髅、孤立圆环及纯色红地面均不产生入口；联合候选落点须来自圆环匹配。"""
        for field in (self.scene(ring=False), self.scene(skull=False), np.full((999, 1776, 3), (0, 0, 255), np.uint8)):
            self.assertEqual(boss_ring_candidates(field), [])
        found = boss_ring_candidates(self.scene())
        self.assertEqual(len(found), 1)
        box = np.asarray(found[0]['box'])
        self.assertLess(np.linalg.norm((box[:2]+box[2:])/2 - [821, 455]), 12.)

    def test_hud_glyph_is_not_a_scene_target(self):
        """完全落入紧凑地图HUD的同一图形不能恢复场景点击许可。"""
        field = self.scene()
        moved = cv2.warpAffine(field, np.float32([[1, 0, -750], [0, 1, -250]]), (1776, 999))
        self.assertEqual(boss_ring_candidates(moved), [])


if __name__ == '__main__':
    unittest.main()
