"""敌人观察簿的坐标失效、分支身份和途中遇敌记账回归。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from module.campaign.scene_memory import SceneEnemyMemory, stable_candidates


class SceneMemoryTests(unittest.TestCase):
    """只有当前可见目标能产生点击候选，旧观察记录本身不能恢复输入许可。"""

    def test_ambiguous_or_shifted_candidates_are_not_clickable(self):
        """红圈消失、镜头大幅移动及近邻歧义均不会产生稳定点击候选。"""
        before = [dict(click=[100., 200.])]
        self.assertEqual(stable_candidates(before, []), [])
        self.assertEqual(stable_candidates(before, [dict(click=[140., 200.])]), [])
        self.assertEqual(stable_candidates(before, [dict(click=[101., 200.]), dict(click=[105., 200.])]), [])
        self.assertEqual(stable_candidates(before, [dict(click=[104., 200.])]), [dict(click=[104., 200.])])

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
            saved = json.loads(path.read_text())
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


if __name__ == '__main__':
    unittest.main()
