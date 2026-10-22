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
        patcher = patch.object(structure, 'facility_catalog', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(structure, 'locate_squad', return_value=dict(
            status='needs_review', identity=[21, 'normal'], map_sha256=None, position=None))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(structure, 'record_map_positions', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(structure, 'nearby_platforms', return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(structure, 'visible_platforms', return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_used_persistent_facility_is_not_clicked_with_a_fresh_input_budget(self):
        """新任务的输入次数归零也不能再次点击当前受阻目标已经试过的持久设施。"""
        target = fixture()
        catalog = Mock()
        catalog.observe.return_value = [dict(target, facility_id='persisted-platform')]
        catalog.eligible.return_value = False
        self.runner.facility_survey_attempted = True
        self.runner.structure_pans = 4
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'detect_objects', return_value=[target]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaises(structure.StructureUnconfirmed):
                structure.recover_visible_structure(self.runner, self.field)
        self.runner.session.tap.assert_not_called()
        catalog.attempt.assert_not_called()

    def test_localization_does_not_preserve_a_stale_facility_click(self):
        """地图复核后设施移动或消失时拒绝旧落点，不能把定位成功当作交互许可。"""
        target = fixture()
        with patch.object(structure, 'detect_objects', side_effect=[
                [target], [target], [fixture(position=(1050., 650.))]]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '旧落点'):
                structure.recover_visible_structure(self.runner, self.field)
        self.runner.session.tap.assert_not_called()
        self.assertEqual(self.runner.structure_inputs, 0)

    def test_nearby_platform_is_not_rejected_as_already_used(self):
        """真实失败中近端距小队约90px的平台优先于峡谷另一侧高分平台，仍只点一次。"""
        near, far = fixture(position=(750., 305.)), fixture(position=(1482., 699.))
        with patch.object(structure, 'detect_objects', return_value=[far]), \
                patch.object(structure, 'nearby_platforms', return_value=[near]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([721., 304.5])), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'attempted')
        self.runner.session.tap.assert_called_once_with(near['position'])
        self.runner.guided_quests.discard.assert_called_once()
        self.assertEqual(self.runner.marker_pans, 0)

    def test_explicit_unanchored_recovery_clicks_one_confirmed_platform_without_pan(self):
        """稳定任务标记可请求原生点击唯一平台，缺小队锚点仍不发送镜头或认领传送。"""
        target = fixture()
        with patch.object(structure, 'detect_objects', return_value=[target]), \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field), \
                patch.object(structure, 'pan_scene') as pan:
            self.assertEqual(structure.recover_visible_structure(
                self.runner, self.field, allow_unanchored=True), 'attempted')
        self.runner.session.tap.assert_called_once_with(target['position'])
        pan.assert_not_called()
        self.runner.memory.victory.assert_not_called()
        attempts = [call for call in self.runner.session.record_evidence.call_args_list
                    if call.args[0] == 'structure_attempt']
        self.assertEqual(attempts[0].kwargs['squad_anchor'], 'unobserved')

    def test_unanchored_ambiguity_or_spent_budget_never_sends_input(self):
        """无锚点时多设施和已耗尽的两次输入均不得退化为猜距离或拖镜头。"""
        for objects, spent in (([fixture(), fixture(position=(650., 500.))], 0), ([fixture()], 2)):
            self.runner.structure_inputs = spent
            with patch.object(structure, 'detect_objects', return_value=objects), \
                    patch.object(structure, 'squad_arrow', return_value=None), \
                    patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                    patch.object(structure, 'pan_scene') as pan:
                with self.assertRaises(structure.StructureUnconfirmed):
                    structure.recover_visible_structure(self.runner, self.field, allow_unanchored=True)
            self.runner.session.tap.assert_not_called()
            pan.assert_not_called()

    def test_unanchored_transient_miss_still_requires_two_adjacent_confirmations(self):
        """首帧平台短暂漏检时最多补两帧；两次连续高分后才发唯一输入。"""
        target = fixture()
        with patch.object(structure, 'detect_objects', side_effect=[[], [target], [target], [target]]) as detect, \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            self.assertEqual(structure.recover_visible_structure(
                self.runner, self.field, allow_unanchored=True), 'attempted')
        self.assertEqual(detect.call_count, 4)
        self.runner.session.tap.assert_called_once_with(target['position'])

    def test_unanchored_permanent_miss_stops_after_four_observations(self):
        """持续漏检不能把短时多帧观测变成无限等待，更不能据此猜点击点。"""
        with patch.object(structure, 'detect_objects', return_value=[]) as detect, \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'pan_scene') as pan:
            with self.assertRaises(structure.StructureUnconfirmed):
                structure.recover_visible_structure(self.runner, self.field, allow_unanchored=True)
        self.assertEqual(detect.call_count, 4)
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
        clock = [0.]
        self.runner.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + 1.)
        samples = [dict(state=state) for state in ('unknown', 'stationary', 'stationary', 'stationary')]
        with patch.object(structure.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(structure, 'MinimapStability') as mini, \
                patch.object(structure, 'NativeMotion') as motion, \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            mini.return_value.observe.side_effect = samples
            motion.return_value.observe.return_value = dict(state='unknown')
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
        self.assertEqual(self.runner.map_frame.call_count, 4)
        self.runner.session.tap.assert_not_called()

    def test_missing_minimap_times_out_with_raw_evidence(self):
        """小地图一直缺小队时按单调时钟退出，保存原图，禁止继续发目标。"""
        clock = [0.]
        self.runner.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        with patch.object(structure.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(structure, 'MinimapStability') as mini, \
                patch.object(structure, 'NativeMotion') as motion, \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            mini.return_value.observe.return_value = dict(state='unknown')
            motion.return_value.observe.return_value = dict(state='unknown')
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '未确认'):
                structure.wait_structure_settle(self.runner, self.field)
        self.assertLessEqual(self.runner.map_frame.call_count, 40)
        self.runner.session.tap.assert_not_called()
        self.assertEqual(self.runner.session.record_evidence.call_args.args[0], 'structure_wait_timeout')


    def test_unobserved_squad_recenters_once_then_waits_again(self):
        """实录22-36：点远处开关后一直看不到小队；找回镜头一次后重新等停稳，仍看不到才停止。"""
        clock = [0.]
        self.runner.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        unobserved = dict(state='unknown', reason='squad_unobserved')
        with patch.object(structure.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(structure, 'MinimapStability') as mini, \
                patch.object(structure, 'NativeMotion') as motion, \
                patch.object(structure, 'recenter_squad') as recenter, \
                patch.object(structure, 'squad_arrow', return_value=None) as arrow, \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            mini.return_value.observe.return_value = dict(state='unknown')
            motion.return_value.observe.return_value = unobserved
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '未确认'):
                structure.wait_structure_settle(self.runner, self.field)
            recenter.assert_called_once()
            recenter.side_effect = lambda runner: arrow.configure_mock(return_value=np.array([888., 400.]))
            recenter.reset_mock()
            motion.return_value.observe.side_effect = lambda field, squad: (
                unobserved if squad is None else dict(state='stationary'))
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
            recenter.assert_called_once()
        self.runner.session.tap.assert_not_called()


if __name__ == '__main__':
    unittest.main()
