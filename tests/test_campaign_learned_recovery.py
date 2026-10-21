"""学习型原型的实例隔离、旧帧拒绝和镜头/小队运动区分，不加载模型。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign.vision import learned_recovery as learned


class LearnedRecoveryTests(unittest.TestCase):
    """模型输出不能跨实例执行，也不能把相机运动计作实际通行。"""

    def test_default_disabled_and_instance_isolation(self):
        """未显式开启或实例不符时，不进入原型，日常不受影响。"""
        runner = SimpleNamespace(owner=SimpleNamespace(config=SimpleNamespace(config_name='AutoPush')))
        with tempfile.TemporaryDirectory() as folder, patch.object(learned, 'ROOT', Path(folder)):
            self.assertFalse(learned.enabled(runner))
            Path(folder, 'prototype.json').write_text(json.dumps({'enabled': True, 'instance': 'AutoPush'}))
            self.assertTrue(learned.enabled(runner))
            runner.owner.config.config_name = 'DailyFull'
            self.assertFalse(learned.enabled(runner))

    def test_camera_only_does_not_confirm_movement(self):
        """背景与小队同移200px时，小队在世界中没移动，三帧停稳也不能确认通行。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        runner = SimpleNamespace(owner=SimpleNamespace(story_wins=set()), session=Mock(), map_frame=Mock(return_value=field))
        with tempfile.TemporaryDirectory() as folder, patch.object(learned, 'ROOT', Path(folder)):
            recovery = learned.Recovery(runner)
        recovery.snapshot = Mock(return_value=('new.npz', {'ground': [700., 500.]}))
        recovery.request = Mock(return_value={'accepted': True, 'matrix': [[1., 0., 200.], [0., 1., 0.], [0., 0., 1.]]})
        with patch.object(learned.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaisesRegex(ValueError, '未确认真实移动'):
                recovery.settle('old.npz', {'ground': [500., 500.]}, {})
        self.assertEqual(recovery.request.call_count, 3)
        runner.session.tap.assert_not_called()

    def test_changed_geometry_invalidates_click(self):
        """推理后场景移动30px，原落点不再获得输入许可。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        runner = SimpleNamespace(owner=SimpleNamespace(story_wins=set()), session=Mock(), map_frame=Mock(return_value=field))
        with tempfile.TemporaryDirectory() as folder, patch.object(learned, 'ROOT', Path(folder)):
            recovery = learned.Recovery(runner)
        recovery.snapshot = Mock(return_value=('new.npz', {'ground': [530., 500.]}))
        recovery.request = Mock(return_value={'accepted': True, 'matrix': [[1., 0., 30.], [0., 1., 0.], [0., 0., 1.]],
                                              'support': [[0., 0.], [1700., 0.], [1700., 900.], [0., 900.]]})
        with patch.object(learned.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaisesRegex(ValueError, '镜头或小队发生变化'):
                recovery.recheck('old.npz', field, [800., 500.], [500., 500.])
        runner.session.tap.assert_not_called()

    def test_unvalidated_surface_never_sends_input(self):
        """服务即使返回移动动作，未通过表面核验也必须在输入前停止。"""
        class Unconfirmed(RuntimeError):
            """隔离异常类型，避免测试加载整套设备和原生日志依赖。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        runner = SimpleNamespace(owner=SimpleNamespace(story_wins=set()), session=Mock())
        with tempfile.TemporaryDirectory() as folder, patch.object(learned, 'ROOT', Path(folder)):
            recovery = learned.Recovery(runner)
        recovery.snapshot = Mock(return_value=('frame.npz', {'ground': [800., 500.]}))
        recovery.request = Mock(return_value=dict(action='move', click=[900., 500.], surface_validated=False))
        with patch.dict('sys.modules', {'module.campaign.structures': SimpleNamespace(StructureUnconfirmed=Unconfirmed)}):
            with self.assertRaisesRegex(Unconfirmed, '语义'):
                recovery.run(field)
        runner.session.tap.assert_not_called()


if __name__ == '__main__':
    unittest.main()
