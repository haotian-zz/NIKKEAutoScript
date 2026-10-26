"""通用设施恢复的两帧、停稳与预算边界；所有点击均为替身，不连接游戏。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2  # noqa: F401  先载入 cv2：patch.dict 退出时会移除块内新载入的模块，cv2 重复加载会报错
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign import structures as structure
    from module.campaign.world import WorldModel


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
        self.runner = Mock(quest=(21, 15), switch_skip=set(), world=WorldModel())
        # 找机关探索的顺序单独验证（test_campaign_explorer）；这里只看画面内设施的交互与预算。
        self.runner.world.put('switch_hunt', 'quest:(21, 15)', True)
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
        """开启机关、低分、左下按钮／右下章节面板和截断框不得变成新落点；底部中间是场景。"""
        good = fixture('ground_mechanism', 'off')
        objects = [good, fixture('ground_mechanism', 'on'), fixture(confidence=.89),
                   fixture(position=(120., 930.)), fixture(position=(1600., 920.)), fixture(position=(1760., 650.))]
        self.assertEqual(len(structure.actionable([fixture(position=(1000., 920.))])), 1)
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
        self.assertEqual(self.runner.world.count('structure_input', reset_on=('win', 'switch')), 1)
        self.assertEqual(self.runner.world.history, ['facility'])
        self.assertIsNone(self.runner.memory.previous)
        self.runner.memory.victory.assert_not_called()
        observed = [call for call in self.runner.session.record_evidence.call_args_list
                    if call.args[0] == 'structure_observed']
        self.assertEqual(observed[0].kwargs['state'], 'unconfirmed')

    def test_missing_squad_ranks_from_view_center_instead_of_stopping(self):
        """停止策略分级：小队箭头和小地图圆环都读不到时按画面中心排序，照常尝试唯一可信设施。"""
        target = fixture()
        with patch.object(structure, 'detect_objects', return_value=[target]), \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure, 'minimap_squad_ground', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field), \
                patch.object(structure, 'pan_scene') as pan:
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'attempted')
        self.runner.session.tap.assert_called_once_with(target['position'])
        pan.assert_not_called()

    def test_used_persistent_facility_is_not_clicked_with_a_fresh_input_budget(self):
        """新任务的输入次数归零也不能再次点击当前受阻目标已经试过的持久设施。"""
        target = fixture()
        catalog = Mock(data=dict(facilities={}))
        catalog.observe.return_value = [dict(target, facility_id='persisted-platform')]
        catalog.eligible.return_value = False
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'detect_objects', return_value=[target]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            with self.assertRaises(structure.StructureUnconfirmed):
                structure.recover_visible_structure(self.runner, self.field)
        self.runner.session.tap.assert_not_called()
        catalog.attempt.assert_not_called()

    def test_localization_does_not_preserve_a_stale_facility_click(self):
        """地图复核后设施移动或消失时拒绝旧落点，不能把定位成功当作交互许可；下一轮重新观察。"""
        target = fixture()
        with patch.object(structure, 'detect_objects', side_effect=[
                [target], [target], [fixture(position=(1050., 650.))]]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'scouted')
        self.runner.session.tap.assert_not_called()
        self.assertEqual(self.runner.world.count('structure_input', reset_on=('win', 'switch')), 0)

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
        self.assertEqual(self.runner.world.history, ['facility'])

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

    def test_spent_budget_never_sends_input(self):
        """每次真实进展前两次设施交互用完后，无论有没有小队锚点都不再点击或拖镜头。"""
        for _ in range(2):
            self.runner.world.bump('structure_input', reset_on=('win', 'switch'))
        with patch.object(structure, 'detect_objects', return_value=[fixture()]), \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'pan_scene') as pan:
            with self.assertRaises(structure.StructureUnconfirmed):
                structure.recover_visible_structure(self.runner, self.field, allow_unanchored=True)
        self.runner.session.tap.assert_not_called()
        pan.assert_not_called()

    def test_missing_facility_stops_after_hunt_without_guessing(self):
        """画面内没有可信设施、本道路状态已探索过：停止，不靠猜点击点。"""
        with patch.object(structure, 'detect_objects', return_value=[]) as detect, \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'pan_scene') as pan:
            with self.assertRaises(structure.StructureUnconfirmed):
                structure.recover_visible_structure(self.runner, self.field, allow_unanchored=True)
        self.assertEqual(detect.call_count, 2)
        self.runner.session.tap.assert_not_called()
        pan.assert_not_called()

    def test_exhausted_budget_stops_and_does_not_click_again(self):
        """两次设施和四次视角均已用完时，明确停止而不继续同一设施。"""
        for _ in range(2):
            self.runner.world.bump('structure_input', reset_on=('win', 'switch'))
        with patch.object(structure, 'detect_objects', return_value=[fixture()]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.):
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '预算'):
                structure.recover_visible_structure(self.runner, self.field)
        self.runner.session.tap.assert_not_called()

    def catalog(self, facilities, observed):
        """目录替身：persisted 编号已试过，其余没试过；observe 依次返回各次登记结果。"""
        catalog = Mock(data=dict(facilities=facilities))
        catalog.eligible.side_effect = lambda identifier, goal: identifier != 'persisted'
        catalog.project_facility.return_value = None
        catalog.observe.side_effect = observed
        return catalog

    def test_catalog_dedupes_facilities_instead_of_input_cap(self):
        """目录可用时按设施去重：之前已交互两次，画面内没试过的平台照样尝试一次。"""
        target = fixture()
        for _ in range(2):
            self.runner.world.bump('structure_input', reset_on=('win', 'switch'))
        catalog = self.catalog({}, [[dict(target, facility_id='scene-0001')]])
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'detect_objects', return_value=[target]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'attempted')
        self.runner.session.tap.assert_called_once_with(target['position'])
        catalog.attempt.assert_called_once_with('scene-0001', 'quest:(21, 15)')

    def test_known_platform_is_tried_when_view_has_no_untried_facility(self):
        """实录 2026-10-25（HARD 9）：画面里唯一平台以前试过，目录里还有没试过的平台：移镜头过去新鲜确认后点它。"""
        used, pad = fixture(), fixture(position=(700., 520.))
        facilities = {'persisted': dict(kind='elevator', map_position=[100., 100.]),
                      'scene-0004': dict(kind='elevator', map_position=[400., 300.]),
                      'scene-0009': dict(kind='elevator', map_position=[900., 900.]),
                      'scene-0011': dict(kind='ground_mechanism', map_position=[120., 100.])}
        catalog = self.catalog(facilities, [[dict(used, facility_id='persisted')],
                                            [dict(pad, facility_id='scene-0004')]])
        seek = Mock(return_value=dict(client=np.array([690., 515.]), field=self.field, chart=None, report=None))
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'detect_objects', side_effect=[[used], [used], [pad], [pad]]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'locate_view', return_value={}), \
                patch.object(structure, 'gaze_position', return_value=np.array([150., 120.])), \
                patch.object(structure, 'seek_map_point', seek), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'attempted')
        self.assertEqual(seek.call_args.args[1].tolist(), [400., 300.])
        self.runner.session.tap.assert_called_once_with(pad['position'])
        catalog.attempt.assert_called_once_with('scene-0004', 'quest:(21, 15)')
        self.assertEqual(self.runner.world.members('known_platform_tried', 'quest:(21, 15)', reset_on=('switch',)),
                         {'scene-0004'})

    def test_known_platform_look_excludes_ghosts_in_the_same_view(self):
        """实录 2026-10-25（HARD 9）：三个目录平台投影在同一屏、都没确认到；一次查看排除全部，下一次去远处的平台。"""
        used, pad = fixture(), fixture(position=(700., 520.))
        facilities = {'persisted': dict(kind='elevator', map_position=[100., 100.]),
                      'scene-0004': dict(kind='elevator', map_position=[160., 130.]),
                      'scene-0007': dict(kind='elevator', map_position=[170., 90.]),
                      'scene-0008': dict(kind='elevator', map_position=[175., 85.]),
                      'scene-0014': dict(kind='elevator', map_position=[900., 600.])}
        catalog = self.catalog(facilities, [[dict(used, facility_id='persisted')], [dict(used, facility_id='persisted')],
                                            [dict(pad, facility_id='scene-0014')]])
        near = dict(client=np.array([830., 580.]), field=self.field, chart='near', report='near')
        far = dict(client=np.array([700., 515.]), field=self.field, chart='far', report='far')
        seek = Mock(side_effect=[near, far])
        in_view = Mock(side_effect=lambda chart, report, spot: chart == 'near' and spot[0] < 200.)
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'detect_objects', side_effect=[[used], [used], [used], [used], [pad], [pad]]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'locate_view', return_value={}), \
                patch.object(structure, 'gaze_position', return_value=np.array([150., 120.])), \
                patch.object(structure, 'seek_map_point', seek), \
                patch.object(structure, 'in_seek_view', in_view), \
                patch.object(structure, 'recenter_squad'), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'attempted')
        self.assertEqual([call.args[1].tolist() for call in seek.call_args_list], [[160., 130.], [900., 600.]])
        self.runner.session.tap.assert_called_once_with(pad['position'])
        catalog.attempt.assert_called_once_with('scene-0014', 'quest:(21, 15)')
        self.assertEqual(self.runner.world.count('known_platform_look', 'quest:(21, 15)', reset_on=('switch',)), 2)

    def test_known_platforms_all_tried_stop_without_moving_camera(self):
        """目录里的平台都试过：不移镜头、不点击，按设施都试过停止。"""
        used = fixture()
        catalog = self.catalog({'persisted': dict(kind='elevator', map_position=[100., 100.])},
                               [[dict(used, facility_id='persisted')]])
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'detect_objects', return_value=[used]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([600., 400.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure, 'seek_map_point') as seek:
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '目录'):
                structure.recover_visible_structure(self.runner, self.field)
        seek.assert_not_called()
        self.runner.session.tap.assert_not_called()

    def transit_run(self, projected, unchanged=False):
        """从山脊点远处平台的替身：等待途中收集一帧，目标平台投影到 projected；返回 (结果, 目录)。"""
        pad = fixture(position=(700., 860.))
        catalog = Mock()
        catalog.project_facility.return_value = None if projected is None else np.array(projected)
        catalog.observe.return_value = [dict(pad, facility_id='scene-0013')]
        # 实录：之前已有一次中途换乘记录，目录不再放行；续点是同一次尝试的延续，不看这个。
        catalog.eligible.return_value = False

        def settle(runner, before=None, recenter=True, samples=None):
            if samples is not None:
                samples.append((self.field, np.array([880., 340.])))
            return self.field
        with patch.object(structure, 'wait_structure_settle', side_effect=settle), \
                patch.object(structure, 'scene_unchanged', return_value=unchanged), \
                patch.object(structure, 'detect_objects', return_value=[pad]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([880., 340.])), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            result = structure.attempt_structure(self.runner, dict(pad, facility_id='scene-0013'), self.field,
                                                 catalog, 'quest:None')
        return result, catalog

    def test_platform_attempt_stopped_short_resumes_once(self):
        """实录 2026-10-25（HARD 9）：小队换乘到楼顶就停，离目标平台约500px：记中途换乘，从新位置续点一次。"""
        result, catalog = self.transit_run([550., 840.])
        self.assertEqual(result, 'attempted')
        self.assertEqual(self.runner.session.tap.call_count, 2)
        self.assertEqual(catalog.attempt.call_count, 2)
        self.assertEqual([call.args + tuple(call.kwargs.values()) for call in catalog.transit.call_args_list],
                         [('scene-0013', True), ('scene-0013', False)])

    def test_platform_reached_or_unmoved_is_not_transit(self):
        """小队到过目标平台旁（坐的就是它）或画面没变，都按一次普通尝试处理，不续点。"""
        result, catalog = self.transit_run([890., 450.])
        self.assertEqual(result, 'attempted')
        catalog.transit.assert_not_called()
        self.runner.session.tap.reset_mock()
        result, catalog = self.transit_run([550., 840.], unchanged=True)
        catalog.project_facility.assert_not_called()
        catalog.transit.assert_not_called()
        self.runner.session.tap.assert_called_once()
        self.runner.session.tap.reset_mock()
        result, catalog = self.transit_run(None)
        catalog.transit.assert_not_called()
        self.runner.session.tap.assert_called_once()

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
        """小地图一直缺小队时按单调时钟退出并保存原图；停止策略分级后交回下一轮重新扫描，不整轮停止。"""
        clock = [0.]
        self.runner.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        with patch.object(structure.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(structure, 'MinimapStability') as mini, \
                patch.object(structure, 'NativeMotion') as motion, \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            mini.return_value.observe.return_value = dict(state='unknown')
            motion.return_value.observe.return_value = dict(state='unknown')
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
        self.assertLessEqual(self.runner.map_frame.call_count, 40)
        self.runner.session.tap.assert_not_called()
        self.assertEqual(self.runner.session.record_evidence.call_args.args[0], 'structure_wait_timeout')


    def test_unobserved_squad_recenters_once_then_waits_again(self):
        """实录22-36：点远处开关后一直看不到小队；找回镜头一次后重新等停稳，仍看不到就交回下一轮。"""
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
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
            recenter.assert_called_once()
            recenter.side_effect = lambda runner: arrow.configure_mock(return_value=np.array([888., 400.]))
            recenter.reset_mock()
            motion.return_value.observe.side_effect = lambda field, squad: (
                unobserved if squad is None else dict(state='stationary'))
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
            recenter.assert_called_once()
        self.runner.session.tap.assert_not_called()

    def test_edge_squad_pans_toward_squad_once_then_waits_again(self):
        """实录23-31：小队停在画面左缘、背景比对覆盖不到；镜头朝小队平移一次后重新等停稳。"""
        clock = [0.]
        self.runner.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0] + seconds)
        outside = dict(state='unknown', reason='squad_outside_background_support')
        with patch.object(structure.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(structure, 'MinimapStability') as mini, \
                patch.object(structure, 'NativeMotion') as motion, \
                patch.object(structure, 'pan_scene') as pan, \
                patch.object(structure, 'squad_arrow', return_value=np.array([75., 480.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            mini.return_value.observe.return_value = dict(state='unknown')
            motion.return_value.observe.return_value = outside
            pan.side_effect = lambda *args, **kwargs: motion.return_value.observe.configure_mock(
                return_value=dict(state='stationary'))
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
            pan.assert_called_once()
            drag = pan.call_args.args[1]
            self.assertGreater(drag[0], 0.)
            self.assertLessEqual(np.linalg.norm(drag), 480.+1e-6)
            pan.reset_mock()
            pan.side_effect = None
            motion.return_value.observe.return_value = outside
            self.assertIs(structure.wait_structure_settle(self.runner, self.field), self.field)
            pan.assert_called_once()
        self.runner.session.tap.assert_not_called()


if __name__ == '__main__':
    unittest.main()
