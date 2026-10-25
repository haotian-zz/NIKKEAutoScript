"""大型入口补充语义的关键反例：骷髅和实际脚圈必须在同一新帧同时成立。"""

from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from module.campaign.scene_memory import follow_bosses, stable_candidates
from module.campaign.vision.boss_ring import boss_ring_candidates, portal_track, quest_portal_candidates, red_shape
from module.campaign.vision.enemy_ring import RingFrame


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

    def test_tilted_hard_boss_skull(self):
        """实录 2026-10-23（困难 5-20 BOSS）：骷髅印在斜转的粉色面片上、外接框竖长，仍须认出并落在地面脚圈。"""
        crop = cv2.imread(str(Path(__file__).resolve().parent / 'fixtures/hard_boss_5_20.png'))
        field = np.full((999, 1776, 3), 40, np.uint8)
        field[260:500, 1110:1350] = crop
        found = boss_ring_candidates(field)
        self.assertEqual(len(found), 1)
        x, y, width, height = found[0]['glyph']
        self.assertLess(width / height, 1.)
        box = np.asarray(found[0]['box'])
        self.assertLess(np.linalg.norm((box[:2]+box[2:])/2 - [1235, 420]), 40.)



class PortalTrackingTests(unittest.TestCase):
    """实录 2026-10-25（HARD 9）：骷髅面片旋转，侧对镜头的帧认不出；已确认的入口按红色形状跟踪。"""

    def portal(self, skull=True, shift=(0, 0)):
        """5-20 困难 Boss 实拍入口放在暗底上；skull=False 时把白色骷髅涂成面片的粉色（模拟侧转）。"""
        crop = cv2.imread(str(Path(__file__).resolve().parent / 'fixtures/hard_boss_5_20.png'))
        if not skull:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            crop = crop.copy()
            crop[(hsv[..., 2] >= 170) & (hsv[..., 1] <= 120)] = (150, 70, 220)
        field = np.full((999, 1776, 3), 40, np.uint8)
        x, y = 1110+shift[0], 260+shift[1]
        field[y:y+240, x:x+240] = crop
        return field

    def frame(self, field=None, box=None):
        """只含 Boss 入口跟踪信息的场景帧（无普通脚圈、无特征；配准在测试里替换）。"""
        tracks = () if box is None else (portal_track(red_shape(field), box),)
        centers = np.empty((0, 2)) if box is None else np.asarray([(np.asarray(box[:2])+box[2:])/2])
        kinds = () if box is None else ('boss',)
        return RingFrame(centers, np.empty((0, 2)), None, np.zeros(2), [], [], kinds, (), (), tracks)

    def test_confirmed_portal_is_followed_when_skull_turns_away(self):
        confirmed = self.portal()
        box = boss_ring_candidates(confirmed)[0]['box']
        turned = self.portal(skull=False, shift=(30, 10))
        self.assertEqual(boss_ring_candidates(turned), [])
        translation = np.float32([[1, 0, 30], [0, 1, 10], [0, 0, 1]])
        with patch('module.campaign.scene_memory.scene_registration', return_value=(translation, None, None)):
            followed = follow_bosses(self.frame(confirmed, box), self.frame(), turned)
        self.assertEqual(followed.kinds, ('boss',))
        expected = (np.asarray(box[:2])+box[2:])/2+[30, 10]
        self.assertLess(np.linalg.norm(followed.centers[0]-expected), 1.)
        self.assertEqual(len(followed.boss_tracks), 1)

    def test_followed_position_is_registration_not_shape_match(self):
        """实录 2026-10-25（HARD 9 山脊）：面片和光柱动画让形状匹配偏 10～13px，落点仍取配准投影，两帧确认通过。"""
        confirmed = self.portal()
        evidence = boss_ring_candidates(confirmed)[0]
        box = evidence['box']
        drifted = self.portal(skull=False, shift=(10, -8))
        with patch('module.campaign.scene_memory.scene_registration', return_value=(np.eye(3), None, None)):
            followed = follow_bosses(self.frame(confirmed, box), self.frame(), drifted)
        center = (np.asarray(box[:2])+box[2:])/2
        self.assertLess(np.linalg.norm(followed.centers[0]-center), 1.)
        x, y, width, height = evidence['glyph']
        before = dict(kind='boss', click=center.tolist(),
                      boss_visual=dict(anchor=[x+width/2, y+height/2], region=evidence['box']))
        after = dict(kind='boss', click=followed.centers[0].tolist())
        self.assertEqual(len(stable_candidates([before], [after])), 1)

    def test_portal_gone_or_registration_failed_is_not_followed(self):
        confirmed = self.portal()
        box = boss_ring_candidates(confirmed)[0]['box']
        empty = np.full((999, 1776, 3), 40, np.uint8)
        with patch('module.campaign.scene_memory.scene_registration', return_value=(np.eye(3), None, None)):
            self.assertEqual(follow_bosses(self.frame(confirmed, box), self.frame(), empty).kinds, ())
        with patch('module.campaign.scene_memory.scene_registration', side_effect=ValueError('场景纹理不足。')):
            self.assertEqual(follow_bosses(self.frame(confirmed, box), self.frame(), self.portal(skull=False)).kinds, ())


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
