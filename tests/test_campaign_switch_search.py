"""受阻后螺旋寻找关闭机关：镜头只用于观察，找到后由设施恢复发送一次新鲜落点。"""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign import structures as structure
    from module.campaign import switch_search as search
# 恢复入口在运行时导入搜索模块；patch.dict退出会移除它，放回同一对象避免重复加载cv2。
sys.modules.setdefault('module.campaign.switch_search', search)


def fixture(kind='ground_mechanism', state='off', position=(1285., 650.), confidence=.95):
    """完整入屏的设施检测框，点击点即框中心。"""
    x, y = position
    return dict(kind=kind, state=state, position=list(position), confidence=confidence,
                box=[x - 40., y - 30., x + 40., y + 30.])


class SwitchSearchTests(unittest.TestCase):
    """21-15实录：门旁只有单向到达平台，开关在南侧多屏之外。"""

    def setUp(self):
        """替身会话与地图帧；检测结果按观察次数注入。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.runner = SimpleNamespace(session=Mock(), quest=(21, 15), map_frame=Mock(return_value=self.field))
        for name, value in (('battle_popup_score', 0.),):
            patcher = patch.object(search.goto, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (('squad_arrow', None), ('ring_centers', []), ('story_markers', []),
                            ('content_shift', None)):
            patcher = patch.object(search, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # 小步航点、增益1、配准不可用时按增益估算：相邻航点恰好一次拖动，便于数手势。
        for name, value in (('SPIRAL_UNIT', np.array([300., 200.])), ('INITIAL_GAIN', 1.),
                            ('WAYPOINT_TOLERANCE', np.array([50., 50.]))):
            patcher = patch.object(search, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_spiral_expands_south_east_then_north_west(self):
        """航点先向南、东各一格，再北、西各两格，逐圈外扩；第一圈8个航点围住起点。"""
        cells = search.spiral_cells()
        self.assertEqual([tuple(cell) for cell in cells[:8]], [
            (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])

    def test_pans_follow_measured_shift_until_waypoint_reached(self):
        """实测22-27：520px竖拖画面只动约0.45倍；按实测位移连拖到离航点不足四分之一屏才取帧。"""
        with patch.object(search, 'SPIRAL_UNIT', np.array([1420., 800.])), \
                patch.object(search, 'INITIAL_GAIN', .45), \
                patch.object(search, 'WAYPOINT_TOLERANCE', np.array([450., 250.])), \
                patch.object(search, 'detect_objects', return_value=[]), \
                patch.object(search, 'pan_scene') as pan:
            search.content_shift.side_effect = lambda before, after: .45*pan.call_args.args[1]
            finder = search.SwitchSearch(self.runner, pans=1)
            finder.move_to(1)
        self.assertEqual(pan.call_count, 3)
        self.assertTrue(all(abs(call.args[1][1]) <= search.VERTICAL_DRAG for call in pan.call_args_list))
        self.assertLessEqual(abs(finder.camera[1]-800.), 250.)

    def test_map_boundary_stops_waypoint_after_two_still_pans(self):
        """连续两次拖动画面几乎不动：到了地图边界，停在原处不再空拖。"""
        with patch.object(search, 'pan_scene') as pan:
            search.content_shift.return_value = np.array([3., -2.])
            finder = search.SwitchSearch(self.runner, pans=1)
            finder.move_to(1)
        self.assertEqual(pan.call_count, 2)

    def test_switch_found_after_pans_needs_two_of_four_frames(self):
        """两帧一致即结束本屏；第三屏单帧闪现补到四帧也不算；第四屏烟雾漏检一帧，四帧三帧命中即返回。"""
        frames = [[]]*6 + [[fixture()], [], [], []] + [[fixture()], [], [fixture()], [fixture(position=(1287., 652.))]]
        with patch.object(search, 'detect_objects', side_effect=frames) as detect, \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner).run()
        self.assertEqual(result['state'], 'found')
        self.assertEqual(result['pans'], 4)
        self.assertEqual(pan.call_count, 4)
        self.assertEqual(detect.call_count, len(frames))
        self.assertEqual(result['target']['position'], [1287., 652.])
        self.assertFalse(self.runner.map_frame.call_args.kwargs['track'])

    def test_on_switch_platform_and_skipped_key_are_not_targets(self):
        """开启机关、平台和本受阻点已试过的搜索键都不再选择，预算耗尽返回exhausted。"""
        seen = [fixture(state='on', position=(400., 400.)), fixture('elevator', 'end', (700., 600.)), fixture()]
        key = (0, *np.round(np.asarray([1285., 650.])/80.).astype(int).tolist())
        with patch.object(search, 'detect_objects', return_value=seen), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner, pans=2, skip={key}).run()
        self.assertEqual(result['state'], 'found')
        self.assertEqual(result['pans'], 1)
        with patch.object(search, 'detect_objects', return_value=seen[:2]), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner, pans=2).run()
        self.assertEqual(result['state'], 'exhausted')
        self.assertEqual(pan.call_count, 2)

    def test_switch_on_same_screen_is_returned_before_gate_enemy(self):
        """机关优先：同屏两帧稳定的敌人和关闭机关，先返回机关。"""
        ring = [(np.array([830., 470.]), .71)]
        with patch.object(search, 'detect_objects', return_value=[fixture()]), \
                patch.object(search, 'ring_centers', return_value=ring), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner).run()
        self.assertEqual(result['state'], 'found')
        pan.assert_not_called()

    def test_gate_enemy_noted_until_every_screen_checked_then_revisited(self):
        """实录21-32/22-27：第1屏的敌人只记下，看完全部屏幕没有机关才逆序倒回第1屏重新确认后返回。"""
        enemy = [(np.array([830., 470.]), .71)]
        rings = {1: enemy}
        screen = iter([0, 0, 1, 1, 2, 2, 3, 3, 1, 1])
        with patch.object(search, 'detect_objects', return_value=[]), \
                patch.object(search, 'ring_centers', side_effect=lambda image: rings.get(next(screen), [])), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner, pans=3).run()
        self.assertEqual(result['state'], 'enemy')
        self.assertEqual(result['step'], 1)
        self.assertTrue(result['complete'])
        drags = [tuple(call.args[1]) for call in pan.call_args_list]
        # 南、东、北三个航点各一拖；倒回时从第3航点(1,0)一次斜拖回第1航点(0,1)。
        self.assertEqual(drags, [(0., -200.), (-300., 0.), (0., 200.), (300., -200.)])

    def test_switch_later_in_spiral_beats_earlier_enemy(self):
        """实录22-27：第1屏是光幕后的首领组，第3屏才有关闭机关；连续看完时直接返回机关，不去打敌人。"""
        enemy = [(np.array([1232., 139.]), .88)]
        frames = [[]]*6 + [[fixture()]]*2
        with patch.object(search, 'detect_objects', side_effect=frames), \
                patch.object(search, 'ring_centers', side_effect=[[], [], enemy, enemy, [], [], [], []]), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner, pans=8).run()
        self.assertEqual(result['state'], 'found')
        self.assertEqual(result['step'], 3)
        self.assertEqual(pan.call_count, 3)

    def test_just_stalled_target_is_not_returned_as_gate_enemy(self):
        """实录21-32：原地第一屏的“守关敌人”(1105,721)就是刚撞墙的目标(1108,721)，须排除。"""
        ring = [(np.array([1105., 721.]), .91)]
        with patch.object(search, 'detect_objects', return_value=[]), \
                patch.object(search, 'ring_centers', return_value=ring), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner, pans=1, blocked=[[1108., 721.]]).run()
        self.assertEqual(result['state'], 'enemy')
        self.assertEqual(pan.call_count, 1)

    def test_ring_under_quest_marker_or_single_frame_is_not_a_gate_enemy(self):
        """任务菱形正下方的脚圈就是受阻首领本身；只出现一帧的脚圈也不算。"""
        boss = (np.array([810., 729.]), .9)
        marker = dict(kind='scene', position=[805., 537.], confidence=.9)
        self.assertEqual(search.gate_enemies([boss], [boss], [marker]), [])
        self.assertEqual(search.gate_enemies([], [boss], []), [])
        self.assertEqual(len(search.gate_enemies([boss], [boss], [dict(marker, kind='edge')])), 1)

    def test_goal_view_ignores_enemies_and_prefers_switch_next_to_quest_marker(self):
        """实录22-21：目标旁的敌人与目标同在光幕后不算守关敌人；两个开关取离任务菱形近的。"""
        near, far = fixture(position=(1041., 467.)), fixture(position=(300., 800.))
        ring = [(np.array([880., 520.]), .9)]
        marker = [dict(kind='scene', position=[900., 420.], confidence=.9)]
        with patch.object(search, 'detect_objects', return_value=[far, near]), \
                patch.object(search, 'ring_centers', return_value=ring), \
                patch.object(search, 'story_markers', return_value=marker), \
                patch.object(search, 'squad_arrow', return_value=np.array([320., 760.])):
            state, target, _ = search.SwitchSearch(self.runner, pans=0).look('goal', goal=True)
        self.assertEqual(state, 'found')
        self.assertEqual(target['position'], [1041., 467.])
        self.assertEqual(target['search_key'][0], 'goal')

    def test_ring_under_blue_cue_marker_is_the_blocked_boss(self):
        """实录22-27第4屏：蓝底菱形识别为cue，正下方(703,460)是首领本身，不当守关敌人。"""
        boss = [(np.array([703., 460.]), .92)]
        marker = dict(kind='cue', position=[713., 185.], confidence=.9)
        with patch.object(search, 'detect_objects', return_value=[]), \
                patch.object(search, 'ring_centers', return_value=boss), \
                patch.object(search, 'story_markers', return_value=[marker]), \
                patch.object(search, 'pan_scene'):
            result = search.SwitchSearch(self.runner, pans=1).run()
        self.assertEqual(result['state'], 'exhausted')

    def test_resumed_spiral_only_pans_through_seen_screens(self):
        """续看第4个航点：按实测位移直接移过去，第0～3屏不取帧，不会再选中那批敌人。"""
        ring = [(np.array([974., 147.]), .88)]
        with patch.object(search, 'detect_objects', return_value=[]) as detect, \
                patch.object(search, 'ring_centers', return_value=ring), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner, pans=8, start=4, enemy_from=4, continuous=False).run()
        self.assertEqual(result['state'], 'enemy')
        self.assertEqual(result['step'], 4)
        self.assertEqual(pan.call_count, 1)
        self.assertEqual(detect.call_count, search.FACILITY_QUICK_FRAMES)

    def test_battle_popup_returns_entry_without_more_pans(self):
        """途中真实备战弹窗立即交回战斗流程。"""
        with patch.object(search.goto, 'battle_popup_score', return_value=.95), \
                patch.object(search, 'detect_objects', return_value=[]), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner).run()
        self.assertEqual(result['state'], 'entry')
        pan.assert_not_called()


class RecoveryOrderTests(unittest.TestCase):
    """门旁只有平台时先找开关；找不到才回到平台，同一受阻目标只搜一次。"""

    def setUp(self):
        """正式恢复入口的替身环境：目录关闭、地图定位不可用。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.runner = Mock(structure_inputs=0, structure_pans=0, quest=(21, 15),
                           switch_searches={}, switch_skip=set(), gate_enemy_tries={})
        self.runner.map_frame.return_value = self.field
        self.runner.session = Mock()
        for name, value in (('facility_catalog', None), ('nearby_platforms', []), ('visible_platforms', []),
                            ('locate_squad', None)):
            patcher = patch.object(structure, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def recover(self, searched, look=None):
        """以门旁到达平台为唯一可见设施执行一次恢复；look注入目标旁那一屏的结果。"""
        pad = fixture('elevator', 'end', (726., 605.))
        finder = Mock()
        finder.return_value.run.return_value = searched
        if look is not None:
            finder.return_value.look = look
        with patch.object(structure, 'detect_objects', return_value=[pad]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([885., 420.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.), \
                patch.object(search, 'SwitchSearch', finder), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            return structure.recover_visible_structure(self.runner, self.field), finder

    def test_switch_found_by_search_is_tapped_instead_of_one_way_pad(self):
        """搜索找到的开关是本次唯一输入，搜索键记为已试，不点门旁平台。"""
        switch = dict(fixture(), search_key=(5, 16, 8))
        result, finder = self.recover(dict(state='found', target=switch, field=self.field, views=6, pans=5))
        self.assertEqual(result, 'attempted')
        self.runner.session.tap.assert_called_once_with(switch['position'])
        self.assertIn((5, 16, 8), self.runner.switch_skip)
        self.assertEqual(self.runner.switch_searches, {'quest:(21, 15)': 1})
        self.assertEqual(self.runner.structure_inputs, 1)

    def test_gate_enemy_is_tapped_and_search_stays_available(self):
        """搜索到守关敌人：点脚圈、按原生寻路等备战；开战后同一受阻目标还能再搜。"""
        enemy = dict(kind='enemy_ring', state='pending', position=[833., 472.], confidence=.72,
                     search_key=('enemy', 3, 10, 6))
        self.runner.wait_entry.return_value = True
        result, _ = self.recover(dict(state='enemy', target=enemy, field=self.field, views=4, pans=3))
        self.assertEqual(result, 'entry')
        self.runner.session.tap.assert_called_once_with([833., 472.])
        self.assertEqual(self.runner.wait_entry.call_args.kwargs['target'], dict(click=[833., 472.]))
        self.assertIn(('enemy', 3, 10, 6), self.runner.switch_skip)
        self.assertEqual(self.runner.switch_searches, {'quest:(21, 15)': 0})
        self.assertEqual(self.runner.structure_inputs, 0)
        self.runner.wait_entry.return_value = False
        self.runner.gate_enemy_tries['quest:(21, 15)'] = 2
        result, finder = self.recover(dict(state='enemy', target=enemy, field=self.field))
        self.assertEqual(result, 'scouted')
        self.assertTrue(finder.call_args.kwargs['enemies'])
        # 实录21-32：光幕后的敌人走不到，第三次也不占搜索次数；之后的搜索只找机关。
        self.assertEqual(self.runner.switch_searches, {'quest:(21, 15)': 0})
        result, finder = self.recover(dict(state='exhausted'))
        self.assertFalse(finder.call_args.kwargs['enemies'])

    def test_exhausted_spiral_escalates_to_chapter_survey_switch(self):
        """实录21-32：螺旋没找到，全章巡检找到未试过的关闭机关，作为本次唯一输入去踩。"""
        catalog = Mock()
        catalog.observe.return_value = []
        self.runner.facility_survey_attempted = False
        switch = dict(fixture(position=(960., 492.)), facility_id='scene-0007')
        survey = Mock(return_value=dict(state='facility_visible', facility_id='scene-0007', target=switch))
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'survey_for_switch', survey):
            result, finder = self.recover(dict(state='exhausted', views=9, pans=8))
        self.assertEqual(finder.call_args.kwargs['pans'], structure.SPIRAL_PANS)
        self.assertEqual(result, 'attempted')
        self.runner.session.tap.assert_called_once_with([960., 492.])
        catalog.attempt.assert_called_once_with('scene-0007', 'quest:(21, 15)')
        self.assertIn(('survey', 'scene-0007'), self.runner.switch_skip)

    def goal_runner(self, focused=True):
        """带任务定位能力的真实受阻目标替身：每个目标只看一次目标旁。"""
        self.runner.goal_views = set()
        self.runner.focus_quest = Mock(return_value=focused)
        return self.runner

    def test_goal_view_switch_is_tried_before_spiral(self):
        """实录22-21：任务定位后目标旁的开关先踩；不占螺旋次数，原位两帧开启即确认。"""
        switch = dict(fixture(position=(1041., 467.)), search_key=('goal', 13, 6))
        self.goal_runner()
        look = Mock(return_value=('found', switch, self.field))
        with patch.object(structure, 'recenter_squad') as recenter, \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'activated')) as watch:
            result, finder = self.recover(dict(state='exhausted'), look)
        self.assertEqual(result, 'attempted')
        finder.return_value.run.assert_not_called()
        self.assertEqual(finder.call_args.kwargs['enemies'], False)
        recenter.assert_not_called()
        self.runner.focus_quest.assert_called_once()
        self.assertEqual(look.call_args.kwargs, dict(goal=True))
        self.runner.session.tap.assert_called_once_with([1041., 467.])
        watch.assert_called_once()
        # 开关确认开启后全部受阻标记清空（含刚踩的搜索键和已看目标旁），按新道路重新判断。
        self.assertEqual(self.runner.switch_searches, {})
        self.assertEqual(self.runner.switch_skip, set())
        self.assertEqual(self.runner.goal_views, set())
        self.assertEqual(self.runner.structure_inputs, 0)

    def test_unconfirmed_switch_keeps_marks(self):
        """点了开关但原位没确认开启：标记保留，同一状态下不重复点同一处。"""
        switch = dict(fixture(position=(1041., 467.)), search_key=('goal', 13, 6))
        self.goal_runner()
        look = Mock(return_value=('found', switch, self.field))
        with patch.object(structure, 'recenter_squad'), \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'unconfirmed')):
            self.recover(dict(state='exhausted'), look)
        self.assertIn(('goal', 13, 6), self.runner.switch_skip)
        self.assertEqual(self.runner.goal_views, {'quest:(21, 15)'})
        self.assertEqual(self.runner.structure_inputs, 1)

    def test_stalled_gate_enemy_resumes_spiral_until_switch_activated(self):
        """实录22-27：整圈看完只剩第3屏敌人且走不到；下次只为找敌人从第4屏续看，踩亮开关后全部标记清空。"""
        enemy = dict(kind='enemy_ring', state='pending', position=[974., 147.], confidence=.88,
                     search_key=('enemy', 3, 12, 2))
        self.runner.spiral_resume = {}
        self.runner.blocked_since = {}
        self.runner.wait_entry.return_value = False
        self.runner.spiral_done = set()
        result, finder = self.recover(dict(state='enemy', target=enemy, field=self.field, step=3, complete=True))
        self.assertEqual(result, 'scouted')
        self.assertEqual(finder.call_args.kwargs['start'], 0)
        self.assertTrue(finder.call_args.kwargs['continuous'])
        self.assertEqual(self.runner.spiral_resume, {'quest:(21, 15)': 4})
        self.assertEqual(self.runner.spiral_done, {'quest:(21, 15)'})
        switch = dict(fixture(position=(400., 300.)), search_key=(6, 5, 4))
        catalog = Mock()
        catalog.observe.return_value = []
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'activated')):
            result, finder = self.recover(dict(state='found', target=switch, field=self.field, step=6))
        self.assertEqual(finder.call_args.kwargs['start'], 4)
        self.assertEqual(finder.call_args.kwargs['enemy_from'], 4)
        self.assertFalse(finder.call_args.kwargs['continuous'])
        self.assertEqual(result, 'attempted')
        self.assertEqual(self.runner.spiral_done, set())
        self.assertEqual(self.runner.spiral_resume, {})
        self.assertEqual(self.runner.gate_enemy_tries, {})
        self.assertEqual(self.runner.switch_skip, set())
        self.assertEqual(self.runner.switch_searches, {})
        catalog.switch_activated.assert_called_once()

    def test_recenter_skipped_when_squad_already_visible(self):
        """实录22-27：任务定位后小队仍在画面内，没有离屏头像可点，不算找回失败。"""
        with patch.object(structure, 'squad_arrow', return_value=np.array([885., 420.])), \
                patch('module.campaign.vision.minimap_occlusion.recenter_offscreen') as offscreen:
            self.assertIsNone(structure.recenter_squad(self.runner))
        offscreen.assert_not_called()

    def test_empty_goal_view_recenters_squad_then_spirals(self):
        """目标旁没有开关：点头像回到小队再螺旋；同一目标下一次受阻不再看目标旁。"""
        self.goal_runner()
        look = Mock(return_value=('none', None, self.field))
        with patch.object(structure, 'recenter_squad') as recenter:
            result, finder = self.recover(dict(state='exhausted'), look)
            self.assertEqual(result, 'scouted')
            finder.return_value.run.assert_called_once()
            recenter.assert_called_once()
            self.runner.switch_searches.clear()
            self.recover(dict(state='exhausted'))
        self.assertEqual(self.runner.focus_quest.call_count, 1)

    def test_switch_watch_confirms_on_state_or_ends_when_switch_leaves_view(self):
        """点开关后盯原位：连续两帧开启才确认；原位连续三帧不见开关即结束观察。"""
        target = fixture(position=(1041., 467.))
        frames = iter([[target], [dict(target, state='on')], [dict(target, state='on', position=[1043., 468.])]])
        with patch.object(structure, 'detect_objects', side_effect=lambda *a: next(frames)), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            self.assertEqual(structure.watch_switch(self.runner, target)[1], 'activated')
            frames = iter([[target], [], [], []])
            self.assertEqual(structure.watch_switch(self.runner, target)[1], 'unconfirmed')

    def test_spiral_budget_covers_every_designed_pan(self):
        """实录22-27：每屏四帧检测约13秒，时间预算须够走完设计的全部拖动，不能在第4屏截断。"""
        self.assertGreaterEqual(structure.SPIRAL_SECONDS, 13. * (structure.SPIRAL_PANS + 1))
        self.assertLess(structure.SPIRAL_SECONDS, structure.BLOCKED_GOAL_SECONDS)

    def test_blocked_goal_past_time_limit_stops_and_reports(self):
        """同一受阻目标超过总时限即停止，不再搜索或点击。"""
        self.runner.blocked_since = {'quest:(21, 15)': 0.}
        with patch.object(structure.time, 'monotonic', return_value=structure.BLOCKED_GOAL_SECONDS+1.):
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '分钟'):
                self.recover(dict(state='exhausted'))
        self.runner.session.tap.assert_not_called()

    def test_exhausted_search_rescans_then_falls_back_to_platform(self):
        """搜索无果不点旧帧平台；下一次受阻不再搜索，改为新鲜确认的平台。"""
        result, _ = self.recover(dict(state='exhausted', views=21, pans=20))
        self.assertEqual(result, 'scouted')
        self.runner.session.tap.assert_not_called()
        result, finder = self.recover(dict(state='exhausted'))
        self.assertEqual(result, 'attempted')
        finder.assert_not_called()
        self.runner.session.tap.assert_called_once_with([726., 605.])


if __name__ == '__main__':
    unittest.main()
