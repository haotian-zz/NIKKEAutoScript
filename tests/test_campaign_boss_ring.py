"""大型入口补充语义的关键反例：骷髅和实际脚圈必须在同一新帧同时成立。"""

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from module.campaign.vision.boss_ring import boss_ring_candidates, quest_portal_candidates


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



class QuestPortalTests(unittest.TestCase):
    """实录22-36：主线入口是任务菱形正下方的红色光圈，模型不出框；只在画面内菱形下方找完整红环。"""

    def marker(self, x, y, kind='scene'):
        return dict(kind=kind, position=[x, y], box=[x-37, y-37, x+37, y+37], confidence=.85)

    def field(self, ring=True):
        field = np.zeros((999, 1776, 3), np.uint8)
        if ring:
            cv2.ellipse(field, (828, 581), (60, 30), 0, 0, 360, (60, 40, 255), 6)
        return field

    def test_ring_below_scene_marker_is_found(self):
        """菱形 (818,379) 下方约 200px 的红环被找到，落点取环中心；没有环或没有菱形都不给候选。"""
        with patch('module.campaign.vision.story_marker.story_markers', return_value=[self.marker(818, 379)]):
            found = quest_portal_candidates(self.field())
            self.assertEqual(len(found), 1)
            box = np.asarray(found[0]['box'])
            self.assertLess(np.linalg.norm((box[:2]+box[2:])/2 - [828, 581]), 15.)
            self.assertEqual(quest_portal_candidates(self.field(ring=False)), [])
        with patch('module.campaign.vision.story_marker.story_markers', return_value=[]):
            self.assertEqual(quest_portal_candidates(self.field()), [])

    def test_edge_pointer_and_offscreen_marker_are_ignored(self):
        """贴边的离屏指示菱形（下方常是红色 EX 指示）和 edge 形态不找入口。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        cv2.ellipse(field, (99, 827), (40, 25), 0, 0, 360, (60, 40, 255), 6)
        with patch('module.campaign.vision.story_marker.story_markers',
                   return_value=[self.marker(89, 637), self.marker(818, 379, kind='edge')]):
            self.assertEqual(quest_portal_candidates(field), [])
            cv2.ellipse(field, (828, 581), (60, 30), 0, 0, 360, (60, 40, 255), 6)
            self.assertEqual(quest_portal_candidates(field), [])

if __name__ == '__main__':
    unittest.main()
