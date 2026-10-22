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
        for name, value in (('squad_arrow', None), ('ring_centers', []), ('story_markers', [])):
            patcher = patch.object(search, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_spiral_expands_south_east_then_north_west(self):
        """先向南、东各一屏，再北、西各两屏，逐圈外扩；手势是手指方向。"""
        drags = search.spiral_drags()
        self.assertEqual(len(drags), 20)
        self.assertEqual([tuple(d) for d in drags[:6]], [
            (0., -search.VERTICAL_DRAG), (-search.HORIZONTAL_DRAG, 0.),
            (0., search.VERTICAL_DRAG), (0., search.VERTICAL_DRAG),
            (search.HORIZONTAL_DRAG, 0.), (search.HORIZONTAL_DRAG, 0.)])

    def test_switch_found_after_pans_needs_two_frames(self):
        """前三屏无开关、第四屏单帧闪现也不算，第五屏两帧稳定才返回。"""
        frames = [[], [], [], [], [], [], [fixture()], [], [fixture()], [fixture(position=(1287., 652.))]]
        with patch.object(search, 'detect_objects', side_effect=frames), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner).run()
        self.assertEqual(result['state'], 'found')
        self.assertEqual(result['pans'], 4)
        self.assertEqual(pan.call_count, 4)
        self.assertEqual(result['target']['position'], [1287., 652.])

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

    def test_uncleared_gate_enemy_is_returned_before_switch(self):
        """实录21-32：光幕外的21-20两帧稳定，搜索交回敌人；同屏机关排在之后。"""
        ring = [(np.array([830., 470.]), .71)]
        with patch.object(search, 'detect_objects', return_value=[fixture()]), \
                patch.object(search, 'ring_centers', side_effect=[ring, [(np.array([833., 472.]), .72)]]), \
                patch.object(search, 'pan_scene') as pan:
            result = search.SwitchSearch(self.runner).run()
        self.assertEqual(result['state'], 'enemy')
        self.assertEqual(result['target']['position'], [833., 472.])
        self.assertEqual(result['target']['search_key'][0], 'enemy')
        pan.assert_not_called()

    def test_ring_under_quest_marker_or_single_frame_is_not_a_gate_enemy(self):
        """任务菱形正下方的脚圈就是受阻首领本身；只出现一帧的脚圈也不算。"""
        boss = (np.array([810., 729.]), .9)
        marker = dict(kind='scene', position=[805., 537.], confidence=.9)
        self.assertEqual(search.gate_enemies([boss], [boss], [marker]), [])
        self.assertEqual(search.gate_enemies([], [boss], []), [])
        self.assertEqual(len(search.gate_enemies([boss], [boss], [dict(marker, kind='edge')])), 1)

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

    def recover(self, searched):
        """以门旁到达平台为唯一可见设施执行一次恢复。"""
        pad = fixture('elevator', 'end', (726., 605.))
        finder = Mock()
        finder.return_value.run.return_value = searched
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
