"""敌人观察簿的坐标失效、分支身份和途中遇敌记账回归。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import cv2

from module.campaign.scene_memory import SceneEnemyMemory, ring_frame, stable_candidates
from module.campaign.vision.detection import Detection
from module.campaign.vision.enemy_ring import scene_ring_candidates
from module.campaign.vision.perception import arrow_template


class SceneMemoryTests(unittest.TestCase):
    """只有当前可见目标能产生点击候选，旧观察记录本身不能恢复输入许可。"""

    def test_squad_overlap_is_not_an_enemy_but_missing_arrow_preserves_unknown_ring(self):
        """真实箭头模板支持排除小队脚圈；箭头缺失时不臆造中心来排除真实敌人。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        template = arrow_template()
        h, w = template.shape
        field[300:300+h, 800:800+w] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        arrow = Detection('scene_squad_arrow', .99, (800., 300., 800.+w, 300.+h))
        own = Detection('scene_enemy_ring', .95, (760., 370., 880., 440.))
        enemy = Detection('scene_enemy_ring', .95, (1000., 370., 1140., 440.))
        with patch('module.campaign.vision.enemy_ring.detect_grid', return_value=[own, enemy, arrow]) as detect:
            rings, excluded = scene_ring_candidates(field)
        self.assertEqual(rings, [enemy])
        self.assertEqual(excluded, (own.box,))
        detect.assert_called_once()
        with patch('module.campaign.vision.enemy_ring.detect_grid', return_value=[own, enemy]):
            self.assertEqual(scene_ring_candidates(field), ([own, enemy], ()))

    def test_ambiguous_or_shifted_candidates_are_not_clickable(self):
        """红圈消失、镜头大幅移动及近邻歧义均不会产生稳定点击候选。"""
        before = [dict(click=[100., 200.])]
        self.assertEqual(stable_candidates(before, []), [])
        self.assertEqual(stable_candidates(before, [dict(click=[140., 200.])]), [])
        self.assertEqual(stable_candidates(before, [dict(click=[101., 200.]), dict(click=[105., 200.])]), [])
        self.assertEqual(stable_candidates(before, [dict(click=[104., 200.])]), [dict(click=[104., 200.])])

    def test_boss_animation_uses_fresh_stable_glyph_and_shared_click_region(self):
        """真实失败的13px中心摆动可由双帧语义内区确认，标记移动或内区冲突仍拒绝。"""
        before = dict(click=[886., 734.], kind='boss',
                      boss_visual=dict(anchor=[875., 600.], region=[800., 680., 972., 788.]))
        after = dict(click=[873., 731.], kind='boss',
                     boss_visual=dict(anchor=[875., 600.], region=[790., 675., 956., 787.]))
        self.assertEqual(stable_candidates([before], [after]), [after])
        moved = dict(after, boss_visual=dict(anchor=[885., 600.], region=after['boss_visual']['region']))
        self.assertEqual(stable_candidates([before], [moved]), [])
        wrong = dict(after, boss_visual=dict(anchor=[875., 600.], region=[1000., 680., 1170., 788.]))
        self.assertEqual(stable_candidates([before], [wrong]), [])

    def test_untrusted_camera_change_creates_new_observation(self):
        """背景跟踪失败时不在同屏相近坐标上继承旧敌人的身份和不可达状态。"""
        with tempfile.TemporaryDirectory() as folder:
            memory = SceneEnemyMemory(Path(folder) / 'enemies.json')
            frame = SimpleNamespace(centers=np.array([[100., 200.]]))
            with patch('module.campaign.scene_memory.ring_frame', return_value=frame), \
                    patch('module.campaign.scene_memory.scene_tracks', side_effect=ValueError('no matches')):
                first = memory.observe(None)[0]
                memory.identify(first['id'], (20, 6, 'B', 1), 'unreachable')
                fresh = memory.observe(None)[0]
            self.assertNotEqual(first['id'], fresh['id'])
            self.assertIsNone(fresh['stage'])
            self.assertEqual(fresh['status'], 'pending')

    def test_tracked_enemy_retains_identity_at_new_screen_position(self):
        """只有经过背景几何验证的连续红圈才能在镜头移动后保留观察 ID。"""
        with tempfile.TemporaryDirectory() as folder:
            memory = SceneEnemyMemory(Path(folder) / 'enemies.json')
            frames = [SimpleNamespace(centers=np.array([[100., 200.]])),
                      SimpleNamespace(centers=np.array([[300., 250.]]))]
            with patch('module.campaign.scene_memory.ring_frame', side_effect=frames), \
                    patch('module.campaign.scene_memory.scene_tracks',
                          return_value=(np.array([[100., 200.]]), np.array([True]))):
                first = memory.observe(None)[0]
                memory.identify(first['id'], (20, 6, 'B', 1), 'encountered_on_route')
                second = memory.observe(None)[0]
            self.assertEqual(first['id'], second['id'])
            self.assertEqual(second['click'], [300., 250.])
            self.assertEqual(second['stage'], [20, 6, 'B', 1])

    def test_route_victory_does_not_clear_requested_enemy(self):
        """走向目标途中打赢其他怪，只记录实际胜利，不删除远端目标或宣称其已清除。"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'enemies.json'
            memory = SceneEnemyMemory(path)
            with patch('module.campaign.scene_memory.ring_frame',
                       return_value=SimpleNamespace(centers=np.array([[100., 200.]]))):
                target = memory.observe(None)[0]
            memory.identify(target['id'], (20, 7), 'encountered_on_route')
            memory.victory((20, 7))
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['confirmed_wins'], [[20, 7]])
            self.assertEqual(saved['enemies'][0]['status'], 'encountered_on_route')
            self.assertEqual(saved['coordinates'], 'client_pixels_per_frame')
            fresh = SceneEnemyMemory(path)
            self.assertEqual(fresh.records, {})
            self.assertIsNone(fresh.previous)

    def test_progress_releases_future_and_blocked_targets(self):
        """通关可能解锁路障，任务推进可能解锁后续关卡；已胜利关卡仍不释放。"""
        with tempfile.TemporaryDirectory() as folder:
            memory = SceneEnemyMemory(Path(folder) / 'enemies.json')
            memory.records = {
                'blocked': dict(stage=None, status='unreachable'),
                'future': dict(stage=[20, 10], status='deferred'),
                'cleared': dict(stage=[20, 9], status='deferred')}
            memory.reconsider((20, 9), {(20, 9)}, after_win=True)
            self.assertEqual(memory.records['blocked']['status'], 'pending')
            self.assertEqual(memory.records['future']['status'], 'deferred')
            memory.reconsider((20, 12), {(20, 9)})
            self.assertEqual(memory.records['future']['status'], 'pending')
            self.assertEqual(memory.records['cleared']['status'], 'deferred')

    def test_hard_map_releases_locked_stage_after_each_win(self):
        """困难图没有任务（quest 为 None）：打赢一关后放回被锁的关卡，已胜利的不放回，没打赢不放回。"""
        with tempfile.TemporaryDirectory() as folder:
            memory = SceneEnemyMemory(Path(folder) / 'enemies.json')
            memory.records = {
                'locked': dict(stage=[9, 25], status='deferred'),
                'cleared': dict(stage=[9, 24], status='deferred')}
            memory.reconsider(None, {(9, 24)})
            self.assertEqual(memory.records['locked']['status'], 'deferred')
            memory.reconsider(None, {(9, 24)}, after_win=True)
            self.assertEqual(memory.records['locked']['status'], 'pending')
            self.assertEqual(memory.records['cleared']['status'], 'deferred')

    def test_ex_classification_survives_ring_deduplication_and_temporary_ocr_miss(self):
        """模型脚圈与文字脚圈同处只计一只，已验证的 EX 不因下一帧 OCR 漏检变成普通目标。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        ring = Detection('scene_enemy_ring', .99, (500., 500., 600., 550.))
        with patch('module.campaign.vision.enemy_ring.detect_grid', return_value=[ring]), \
                patch('module.campaign.vision.ex_ring.scene_ex_rings', return_value=[ring]):
            frame = ring_frame(field, object())
        self.assertEqual(frame.kinds, ('ex',))
        self.assertEqual(len(frame.centers), 1)
        with tempfile.TemporaryDirectory() as folder:
            memory = SceneEnemyMemory(Path(folder) / 'enemies.json')
            first = memory.observe(field, frame=frame)[0]
            with patch('module.campaign.scene_memory.scene_tracks',
                       return_value=(frame.centers, np.array([True]))):
                second = memory.observe(field, frame=SimpleNamespace(centers=frame.centers))[0]
            self.assertEqual(first['id'], second['id'])
            self.assertEqual(second['kind'], 'ex')


if __name__ == '__main__':
    unittest.main()
