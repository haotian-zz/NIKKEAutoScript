"""通用设施恢复的两帧、停稳与预算边界；所有点击均为替身，不连接游戏。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign import structures as structure


def fixture(kind='elevator', state='start', position=(1000., 650.), confidence=.95):
    """创建完整入屏的语义检测框，保持点击点与框中心一致。"""
    x, y = position
    return dict(kind=kind, state=state, position=list(position), confidence=confidence,
                box=[x - 40., y - 30., x + 40., y + 30.])


class StructureTests(unittest.TestCase):
    """没有识别或停稳证据时立即保留未完成，不在移动途中改发其他目标。"""

    def setUp(self):
        """每项场景使用隔离假会话，显式注入识别结果和单调时钟。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.runner = Mock(structure_inputs=0, structure_pans=0)
        self.runner.map_frame.return_value = self.field
        self.runner.session = Mock()

    def test_only_unactivated_switch_or_full_high_confidence_platform_is_actionable(self):
        """开启机关、低分、底栏和截断框不得变成新落点。"""
        good = fixture('ground_mechanism', 'off')
        objects = [good, fixture('ground_mechanism', 'on'), fixture(confidence=.89),
                   fixture(position=(1000., 920.)), fixture(position=(1760., 650.))]
        self.assertEqual(structure.actionable(objects), [good])

    def test_stability_rejects_wrong_role_shift_and_competition(self):
        """start/end变化、大幅位置变化和邻近竞争均不继承上一帧许可。"""
        first = [fixture()]
        self.assertEqual(structure.stable_structures(first, [fixture(state='end')]), [])
        self.assertEqual(structure.stable_structures(first, [fixture(position=(1020., 650.))]), [])
        self.assertEqual(structure.stable_structures(first, [fixture(), fixture(position=(1005., 650.))]), [])
        self.assertEqual(len(structure.stable_structures(first, [fixture(position=(1005., 650.))])), 1)

    def test_interaction_sends_one_current_target_and_does_not_claim_transfer(self):
        """一次设施尝试只发送一个当前落点，清理旧关联，不写成功或胜利。"""
        target = fixture()
        with patch.object(structure, 'detect_objects', return_value=[target]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'attempted')
        self.runner.session.tap.assert_called_once_with(target['position'])
        self.assertEqual(self.runner.structure_inputs, 1)
        self.assertIsNone(self.runner.memory.previous)
        self.runner.memory.victory.assert_not_called()
        observed = [call for call in self.runner.session.record_evidence.call_args_list
                    if call.args[0] == 'structure_observed']
        self.assertEqual(observed[0].kwargs['state'], 'unconfirmed')

    def test_missing_squad_stops_without_pan_or_click(self):
        """设施可信但小队未定位时，不把镜头手势冒险发成走位。"""
        with patch.object(structure, 'detect_objects', return_value=[fixture()]), \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'pan_scene') as pan:
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '小队位置'):
                structure.recover_visible_structure(self.runner, self.field)
        self.runner.session.tap.assert_not_called()
        pan.assert_not_called()

    def test_exhausted_budget_stops_and_does_not_click_again(self):
        """两次设施和四次视角均已用完时，明确停止而不继续同一设施。"""
        self.runner.structure_inputs, self.runner.structure_pans = 2, 4
        with patch.object(structure, 'detect_objects', return_value=[fixture()]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.):
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '预算'):
                structure.recover_visible_structure(self.runner, self.field)
        self.runner.session.tap.assert_not_called()

    def test_moving_squad_needs_three_new_stable_minimap_frames(self):
        """首帧小队仍在动时不提前结束；之后连续三帧稳定才允许恢复扫描。"""
        road = np.zeros((30, 30), np.uint8)
        valid = np.full_like(road, 255)
        samples = [(None, road, valid, point) for point in ((0., 0.), (8., 0.), (8., 0.), (8., 0.))]
        with patch.object(structure, 'minimap_masks', side_effect=samples), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            self.assertIs(structure.wait_structure_settle(self.runner), self.field)
        self.assertEqual(self.runner.map_frame.call_count, 4)
        self.runner.session.tap.assert_not_called()

    def test_missing_minimap_times_out_with_raw_evidence(self):
        """小地图一直缺小队时按单调时钟退出，保存原图，禁止继续发目标。"""
        clock = [0.]
        self.runner.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        with patch.object(structure.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(structure, 'minimap_masks', side_effect=ValueError('no ring')), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '二十秒'):
                structure.wait_structure_settle(self.runner)
        self.assertLessEqual(self.runner.map_frame.call_count, 40)
        self.runner.session.tap.assert_not_called()
        self.assertEqual(self.runner.session.record_evidence.call_args.args[0], 'structure_wait_timeout')


if __name__ == '__main__':
    unittest.main()
