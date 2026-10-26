"""统一镜头探索器与受阻恢复顺序：镜头只用于观察，找到后由调用方发送一次新鲜落点。"""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2  # 先载入 cv2：patch.dict 退出时会移除块内新载入的模块，cv2 重复加载会报错
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign import structures as structure
    from module.campaign import explorer as explore
    from module.campaign.world import WorldModel
# 恢复入口在运行时导入探索模块；patch.dict退出会移除它，放回同一对象避免重复加载cv2。
sys.modules.setdefault('module.campaign.explorer', explore)


def fixture(kind='ground_mechanism', state='off', position=(1285., 650.), confidence=.95):
    """完整入屏的设施检测框，点击点即框中心。"""
    x, y = position
    return dict(kind=kind, state=state, position=list(position), confidence=confidence,
                box=[x - 40., y - 30., x + 40., y + 30.])


def ring(identifier, click, kind='unknown'):
    """观察簿返回的脚圈候选。"""
    return dict(id=identifier, click=list(click), stage=None, status='pending', kind=kind)


def runner_for(field, quest=(21, 15)):
    """替身运行器：观察簿默认一个脚圈都没有，所有候选都可选、没有被拒绝的位置。"""
    return SimpleNamespace(session=Mock(), quest=quest, map_frame=Mock(return_value=field),
                           memory=SimpleNamespace(observe=Mock(return_value=[])),
                           selectable=lambda item: True, drop_rejected=lambda frame, items: items,
                           switch_skip=set(), world=WorldModel())


class ExplorerTests(unittest.TestCase):
    """21-15实录：门旁只有单向到达平台，开关在南侧多屏之外；HARD 9：没有敌人时机关旁一次没踩。"""

    def setUp(self):
        """替身会话与地图帧；检测结果按观察次数注入。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.runner = runner_for(self.field)
        patcher = patch.object(explore.goto, 'battle_popup_score', return_value=0.)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name, value in (('squad_arrow', None), ('story_markers', []), ('content_shift', None),
                            ('facility_frame', None), ('minimap_hints', []), ('minimap_squad_ground', None)):
            patcher = patch.object(explore, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        # 小步航点、增益1、配准不可用时按增益估算：相邻航点恰好一次拖动，便于数手势。
        for name, value in (('SPIRAL_UNIT', np.array([300., 200.])), ('INITIAL_GAIN', 1.),
                            ('WAYPOINT_TOLERANCE', np.array([50., 50.])), ('FOG_MISSES', 99)):
            patcher = patch.object(explore, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def explorer(self, want=('switch', 'enemy'), rings=1, catalog=None, **kwargs):
        """rings=1：原地加 2 个航点（南、东）。"""
        return explore.Explorer(self.runner, want=want, catalog=catalog, rings=rings, **kwargs)

    def test_offscreen_squad_is_predicted_instead_of_detected(self):
        """小队箭头只在推算位置靠近画面时识别；镜头移开一屏多后按实测位移推算为离屏，不再花时间识别。"""
        finder = self.explorer()
        with patch.object(explore, 'squad_arrow', return_value=np.array([888., 600.])) as arrow, \
                patch.object(explore.goto, 'ARROW_GROUND_OFFSET', 90.):
            ground = finder.squad_ground(self.field)
            self.assertEqual(arrow.call_count, 1)
            self.assertEqual(ground.tolist(), [888., 690.])
            finder.camera = np.array([1500., 0.])
            self.assertIsNone(finder.squad_ground(self.field))
            self.assertEqual(arrow.call_count, 1)
            finder.camera = np.array([400., 0.])
            finder.squad_ground(self.field)
            self.assertEqual(arrow.call_count, 2)

    def test_hidden_arrow_keeps_projected_squad(self):
        """实录22-36：小队在穹顶里箭头被挡、小地图圆环也读不到；按最后看到的位置加镜头位移推算避让。"""
        finder = self.explorer()
        finder.squad, finder.squad_camera = np.array([900., 700.]), np.zeros(2)
        finder.camera = np.array([100., 0.])
        np.testing.assert_allclose(finder.squad_ground(self.field), [800., 700.])

    def test_previous_after_features_are_reused_as_next_before(self):
        """连续拖动时上一帧的配准特征直接复用，只为新帧提特征。"""
        frames = [np.zeros((999, 1776, 3), np.uint8) for _ in range(3)]
        self.runner.map_frame = Mock(side_effect=frames[1:])
        finder = self.explorer()
        with patch.object(explore, 'pan_scene') as pan, \
                patch.object(explore, 'facility_frame', side_effect=['f1', 'f2']) as features:
            after, _, _ = finder.pan(np.array([-500., 0.]), frames[0])
            finder.pan(np.array([-500., 0.]), after)
        self.assertEqual(features.call_count, 2)
        calls = explore.content_shift.call_args_list
        self.assertEqual(calls[0].args[3:], (None, 'f1'))
        self.assertEqual(calls[1].args[3:], ('f1', 'f2'))
        self.assertIsNone(pan.call_args.kwargs['settle'])
        self.assertTrue(pan.call_args.kwargs['paced'])

    def test_spiral_expands_south_east_then_north_west(self):
        """航点先向南、东各一格，再北、西各两格，逐圈外扩；第一圈8个航点围住起点。"""
        cells = explore.spiral_cells()
        self.assertEqual([tuple(cell) for cell in cells[:8]], [
            (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)])

    def test_pans_follow_measured_shift_until_waypoint_reached(self):
        """实测22-27：520px竖拖画面只动约0.45倍；按实测位移连拖到离航点不足四分之一屏才取帧。"""
        with patch.object(explore, 'INITIAL_GAIN', .45), \
                patch.object(explore, 'WAYPOINT_TOLERANCE', np.array([450., 250.])), \
                patch.object(explore, 'pan_scene') as pan:
            explore.content_shift.side_effect = lambda before, after, drag, *features: .45*np.asarray(drag)
            finder = self.explorer()
            finder.move_by(np.array([0., 800.]), tolerance=explore.WAYPOINT_TOLERANCE)
        self.assertEqual(pan.call_count, 3)
        self.assertTrue(all(abs(call.args[1][1]) <= explore.VERTICAL_DRAG for call in pan.call_args_list))
        self.assertLessEqual(abs(finder.camera[1]-800.), 250.)

    def test_map_boundary_stops_waypoint_after_two_still_pans(self):
        """连续两次拖动画面几乎不动：到了地图边界，停在原处不再空拖。"""
        with patch.object(explore, 'pan_scene') as pan:
            explore.content_shift.return_value = np.array([3., -2.])
            finder = self.explorer()
            self.assertFalse(finder.move_by(np.array([0., 600.])))
        self.assertEqual(pan.call_count, 2)
        self.assertEqual(finder.edges, 1)

    def test_fog_pans_back_and_marks_edge(self):
        """实录 HARD 9：拖进悬崖外的雾里连续测不到位移；拖回一格当边界，不整轮中止。"""
        with patch.object(explore, 'FOG_MISSES', 2), patch.object(explore, 'pan_scene') as pan:
            finder = self.explorer()
            self.assertFalse(finder.move_by(np.array([-2000., 0.])))
        drags = [call.args[1].tolist() for call in pan.call_args_list]
        self.assertEqual(len(drags), 3)
        np.testing.assert_allclose(drags[2], -np.asarray(drags[1]))
        self.assertEqual(finder.edges, 1)
        events = [call.args[0] for call in self.runner.session.record_evidence.call_args_list]
        self.assertIn('explore_unmeasured', events)

    def test_switch_found_after_pans_needs_two_of_four_frames(self):
        """两帧一致即结束本屏；单帧闪现补到四帧也不算；烟雾漏检一帧，四帧三帧命中即返回。"""
        frames = [[]]*6 + [[fixture()], [], [], []] + [[fixture()], [], [fixture()], [fixture(position=(1287., 652.))]]
        with patch.object(explore, 'detect_objects', side_effect=frames) as detect, \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer(rings=2).run()
        self.assertEqual(result['state'], 'switch')
        self.assertEqual(pan.call_count, 4)
        self.assertEqual(detect.call_count, len(frames))
        self.assertEqual(result['target']['position'], [1287., 652.])
        self.assertFalse(self.runner.map_frame.call_args.kwargs['track'])

    def test_on_switch_platform_and_skipped_key_are_not_targets(self):
        """开启机关、平台和本受阻点已试过的搜索键都不再选择，航点看完返回exhausted。"""
        seen = [fixture(state='on', position=(400., 400.)), fixture('elevator', 'end', (700., 600.)), fixture()]
        key = ('switch', *np.round(np.asarray([1285., 650.])/80.).astype(int).tolist(), 0, 0)
        with patch.object(explore, 'detect_objects', return_value=seen), patch.object(explore, 'pan_scene'):
            result = self.explorer(skip={key}).run()
        self.assertEqual(result['state'], 'switch')
        self.assertEqual(result['step'], 1)
        with patch.object(explore, 'detect_objects', return_value=seen[:2]), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual(result['state'], 'exhausted')
        self.assertEqual(pan.call_count, 2)

    def test_switch_on_same_screen_is_returned_before_gate_enemy(self):
        """机关优先：同屏两帧稳定的敌人和关闭机关，先返回机关。"""
        self.runner.memory.observe.return_value = [ring('ring-1', (830., 470.))]
        with patch.object(explore, 'detect_objects', return_value=[fixture()]), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual(result['state'], 'switch')
        pan.assert_not_called()

    def test_gate_enemy_noted_until_every_screen_checked_then_revisited(self):
        """实录21-32/22-27：第1屏的敌人只记下，看完全部屏幕没有机关才倒回第1屏重新确认后返回。"""
        screens = iter([0, 1, 1, 2, 1, 1])
        self.runner.memory.observe.side_effect = lambda field: (
            [ring('ring-1', (830., 470.))] if next(screens) == 1 else [])
        with patch.object(explore, 'detect_objects', return_value=[]), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual(result['state'], 'enemy')
        self.assertEqual(result['step'], 1)
        drags = [tuple(call.args[1]) for call in pan.call_args_list]
        # 南、东两个航点各一拖；倒回时从东(1,1)直接横拖回南(0,1)。
        self.assertEqual(drags, [(0., -200.), (-300., 0.), (300., 0.)])

    def observe_screens(self, screens):
        """观察簿按调用次数返回：1 是一个稳定脚圈，0 是没有；没有脚圈的一屏只取一帧。"""
        screens = iter(screens)
        self.runner.memory.observe.side_effect = lambda field: (
            [ring('ring-1', (830., 470.))] if next(screens) == 1 else [])

    def test_rewind_corrects_estimate_by_matching_the_noted_view(self):
        """实录 2026-10-25（HARD 9）：估算倒回后没认出敌人；按当时画面配准还差 120px，补移一次后认出并返回。"""
        self.observe_screens([0, 1, 1, 0, 0, 1, 1])
        with patch.object(explore, 'detect_objects', return_value=[]), \
                patch.object(explore, 'view_offset', return_value=np.array([120., 0.])) as offset, \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual((result['state'], result['step']), ('enemy', 1))
        offset.assert_called_once()
        drags = [tuple(call.args[1]) for call in pan.call_args_list]
        self.assertEqual(drags, [(0., -200.), (-300., 0.), (300., 0.), (-120., 0.)])

    def test_rewind_tries_latest_sighting_first_then_earlier_one(self):
        """第 1、2 屏都见过敌人：先倒回最近的第 2 屏，配不上当时画面就放弃它，再倒回第 1 屏认出后返回。"""
        self.observe_screens([0, 1, 1, 1, 1, 0, 1, 1])
        with patch.object(explore, 'detect_objects', return_value=[]), \
                patch.object(explore, 'view_offset', return_value=None), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual((result['state'], result['step']), ('enemy', 1))
        drags = [tuple(call.args[1]) for call in pan.call_args_list]
        self.assertEqual(drags, [(0., -200.), (-300., 0.), (300., 0.)])

    def test_sighting_with_squad_rewinds_by_recentering_before_newer_ones(self):
        """实录 2026-10-25（HARD 9）：起点那屏看得到小队也看得到敌人；倒回时先点头像找回小队，不按累计估算拖镜头。"""
        self.observe_screens([1, 1, 0, 1, 1, 1, 1])
        arrows = [np.array([888., 600.])]
        recenter = Mock(return_value='marker')
        with patch.object(explore, 'detect_objects', return_value=[]), \
                patch.object(explore, 'squad_arrow', side_effect=lambda frame: arrows.pop() if arrows else None), \
                patch.dict(sys.modules, {'module.campaign.vision.minimap_occlusion':
                                         SimpleNamespace(recenter_offscreen=recenter)}), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual((result['state'], result['step']), ('enemy', 0))
        recenter.assert_called_once()
        self.assertEqual([tuple(call.args[1]) for call in pan.call_args_list], [(0., -200.), (-300., 0.)])

    def test_note_keeps_latest_squad_sighting_and_latest_others(self):
        """看得到小队的记录只留最近一处、不被后来的挤掉；其余只留最近 REWIND_SIGHTINGS 处。"""
        finder, sightings = self.explorer(), []
        for step, squad in enumerate([True, True, False, False, False, False]):
            sightings = finder.note(sightings, dict(field=self.field, squad=np.zeros(2) if squad else None), step)
        self.assertEqual([(item['step'], item['squad']) for item in sightings],
                         [(1, True), (3, False), (4, False), (5, False)])

    def test_enemy_first_returns_rings_immediately(self):
        """没有可点敌人时的探索：敌人优先，第一屏见到稳定脚圈即交回主循环。"""
        self.runner.memory.observe.return_value = [ring('ring-1', (830., 470.))]
        with patch.object(explore, 'detect_objects', return_value=[fixture()]), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer(want=('enemy', 'switch')).run()
        self.assertEqual(result['state'], 'enemy')
        self.assertEqual(result['targets'][0]['id'], 'ring-1')
        pan.assert_not_called()

    def test_switch_only_is_revisited_when_no_enemy_found(self):
        """实录 HARD 9：画面里没有敌人，小队旁的关闭机关已登记却一次没踩；看完没有敌人就回去踩机关。"""
        frames = iter([[fixture()], [fixture()], [], [], [], []])
        with patch.object(explore, 'detect_objects', side_effect=lambda *args: next(frames, [fixture()])), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer(want=('enemy', 'switch')).run()
        self.assertEqual(result['state'], 'switch')
        self.assertEqual(result['step'], 0)
        self.assertEqual(pan.call_count, 3)

    def test_single_empty_frame_skips_second_ring_observation(self):
        """单帧一个脚圈都没有就不取第二帧；有脚圈时两帧唯一对应才算候选。"""
        finder = self.explorer(want=('enemy',))
        _, enemies = finder.enemies(self.field)
        self.assertEqual(enemies, [])
        self.assertEqual(self.runner.memory.observe.call_count, 1)
        self.runner.memory.observe.side_effect = [[ring('a', (800., 500.))], [ring('a', (900., 500.))]]
        _, enemies = finder.enemies(self.field)
        self.assertEqual(enemies, [])
        self.assertEqual(self.runner.memory.observe.call_count, 3)

    def test_goal_view_ignores_enemies_and_prefers_switch_next_to_quest_marker(self):
        """实录22-21：目标旁的敌人与目标同在光幕后不算守关敌人；两个开关取离任务菱形近的。"""
        near, far = fixture(position=(1041., 467.)), fixture(position=(300., 800.))
        self.runner.memory.observe.return_value = [ring('ring-1', (880., 520.))]
        marker = [dict(kind='scene', position=[900., 420.], confidence=.9)]
        with patch.object(explore, 'detect_objects', return_value=[far, near]), \
                patch.object(explore, 'story_markers', return_value=marker), \
                patch.object(explore, 'squad_arrow', return_value=np.array([320., 760.])):
            seen = self.explorer().look('goal', goal=True)
        self.assertEqual(seen['enemies'], [])
        self.assertEqual(seen['switches'][0]['position'], [1041., 467.])
        self.runner.memory.observe.assert_not_called()

    def test_catalog_hint_is_centered_then_confirmed_by_fresh_detection(self):
        """目录记得的关闭机关在画面上部被挡住：先移到画面中下部，新鲜两帧检测到才作为机关返回。"""
        catalog = Mock()
        catalog.observe.return_value = []
        catalog.project_facilities.return_value = {}
        catalog.eligible.return_value = True
        catalog.expected = [dict(facility_id='scene-0003', kind='ground_mechanism', state='off',
                                 position=[1200., 340.])]
        frames = [[], [], [fixture(position=(890., 640.))], [fixture(position=(890., 640.))]]
        with patch.object(explore, 'detect_objects', side_effect=frames), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer(catalog=catalog).run()
        self.assertEqual(result['state'], 'switch')
        self.assertEqual(result['target']['position'], [890., 640.])
        drag = pan.call_args_list[0].args[1]
        self.assertLess(drag[0], 0.)
        self.assertGreater(drag[1], 0.)

    def test_minimap_hint_outside_view_is_chased_first(self):
        """实录 HARD 9：巡检多屏「场景目标 0、小地图提示 3」；画面外的小地图敌人先把镜头移过去看。"""
        hints = iter([[np.array([1700., 900.])], []])
        self.runner.memory.observe.side_effect = [[], [ring('ring-9', (900., 600.))], [ring('ring-9', (900., 600.))]]
        with patch.object(explore, 'minimap_hints', side_effect=lambda field: next(hints, [])), \
                patch.object(explore, 'detect_objects', return_value=[]), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer(want=('enemy', 'switch')).run()
        self.assertEqual(result['state'], 'enemy')
        drag = pan.call_args_list[0].args[1]
        self.assertLess(drag[0], 0.)
        self.assertLess(drag[1], 0.)

    def test_battle_popup_returns_entry_without_more_pans(self):
        """途中真实备战弹窗立即交回战斗流程。"""
        with patch.object(explore.goto, 'battle_popup_score', return_value=.95), \
                patch.object(explore, 'detect_objects', return_value=[]), \
                patch.object(explore, 'pan_scene') as pan:
            result = self.explorer().run()
        self.assertEqual(result['state'], 'entry')
        pan.assert_not_called()


class ContentShiftTests(unittest.TestCase):
    """镜头主平移：顺手指方向的匹配取中位数；大多数不动判为边界；方向不符测不到。"""

    @classmethod
    def setUpClass(cls):
        """固定随机纹理，平移180px模拟一次横向拖动后的画面。"""
        rng = np.random.default_rng(791)
        gray = cv2.resize(rng.integers(0, 256, (500, 888), dtype=np.uint8), (1776, 999))
        cls.field = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cls.panned = cv2.warpAffine(cls.field, np.float32([[1, 0, 180], [0, 1, 0]]), (1776, 999))

    def test_shift_follows_drag_direction(self):
        np.testing.assert_allclose(explore.content_shift(self.field, self.panned, [400., 0.]), [180., 0.], atol=3.)

    def test_still_frame_is_zero_shift_for_boundary_detection(self):
        np.testing.assert_allclose(explore.content_shift(self.field, self.field, [400., 0.]), [0., 0.])

    def test_view_offset_needs_no_drag_direction(self):
        """倒回配准不知道镜头往哪走：两屏位移取最大一簇；配不上或缺特征返回 None。"""
        reference = explore.facility_frame(self.field, [])
        np.testing.assert_allclose(explore.view_offset(reference, explore.facility_frame(self.panned, [])),
                                   [180., 0.], atol=3.)
        np.testing.assert_allclose(explore.view_offset(explore.facility_frame(self.panned, []), reference),
                                   [-180., 0.], atol=3.)
        self.assertIsNone(explore.view_offset(reference, explore.facility_frame(np.zeros_like(self.field), [])))
        self.assertIsNone(explore.view_offset(None, reference))

    def test_motion_against_drag_is_not_measured(self):
        self.assertIsNone(explore.content_shift(self.field, self.panned, [-400., 0.]))


class SceneRegistrationTests(unittest.TestCase):
    """独立合成纹理和已知真实平移验证生产背景配准及红圈跟踪。"""

    def test_known_camera_translation_tracks_rings(self):
        """几何真值来自合成图变换，检测特征及关联使用真实 SIFT 和生产实现。"""
        from module.campaign.vision.enemy_ring import RingFrame, scene_registration, scene_tracks
        rng = np.random.default_rng(371)
        image = cv2.GaussianBlur(rng.integers(0, 256, (900, 1600), dtype=np.uint8), (5, 5), 1.)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 35], [0, 1, -20]]), (1600, 900))
        frames = []
        for field, centers in ((image, [[700, 500], [1050, 650]]),
                               (shifted, [[735, 480], [1085, 630]])):
            keys, descriptors = cv2.SIFT_create(nfeatures=1800).detectAndCompute(field, None)
            frames.append(RingFrame(np.asarray(centers, float), np.asarray([key.pt for key in keys]),
                                    descriptors, np.zeros(2), [], []))
        matrix, _, _ = scene_registration(*frames)
        expected = np.float32([[[400, 400], [800, 500], [1200, 700]]])
        actual = cv2.perspectiveTransform(expected, matrix)
        np.testing.assert_allclose(actual, expected + [35, -20], atol=.5)
        old, tracked = scene_tracks(*frames)
        self.assertTrue(tracked.all())
        np.testing.assert_allclose(old, frames[0].centers, atol=.5)


class RecoveryOrderTests(unittest.TestCase):
    """门旁只有平台时先找开关；找不到才回到平台，同一道路状态下每个受阻目标只探索一次。"""

    def setUp(self):
        """正式恢复入口的替身环境：目录关闭、地图定位不可用。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.runner = Mock(quest=(21, 15), switch_skip=set(), world=WorldModel())
        self.runner.map_frame.return_value = self.field
        self.runner.session = Mock()
        for name, value in (('facility_catalog', None), ('nearby_platforms', []), ('visible_platforms', []),
                            ('locate_squad', None), ('minimap_squad_ground', None)):
            patcher = patch.object(structure, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def recover(self, searched, look=None):
        """以门旁到达平台为唯一可见设施执行一次恢复；look 注入只找机关地看一屏的结果。"""
        pad = fixture('elevator', 'end', (726., 605.))
        finder = Mock()
        finder.return_value.run.return_value = searched
        with patch.object(structure, 'detect_objects', return_value=[pad]), \
                patch.object(structure, 'squad_arrow', return_value=np.array([885., 420.])), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(structure.goto, 'ARROW_GROUND_OFFSET', 90.), \
                patch.object(explore, 'Explorer', finder), \
                patch.object(structure, 'switch_look', look or Mock(return_value=('none', None, self.field))), \
                patch.object(structure, 'wait_structure_settle', return_value=self.field):
            return structure.recover_visible_structure(self.runner, self.field), finder

    def test_switch_found_by_explorer_is_tapped_instead_of_one_way_pad(self):
        """探索找到的开关是本次唯一输入，搜索键记为已试，不点门旁平台。"""
        switch = dict(fixture(), search_key=('switch', 16, 8, 5, 0))
        with patch.object(structure, 'recenter_squad'), \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'unconfirmed')):
            result, finder = self.recover(dict(state='switch', target=switch, field=self.field, views=6, pans=5))
        self.assertEqual(result, 'attempted')
        self.assertEqual(finder.call_args.kwargs['want'], ('switch', 'enemy'))
        self.runner.session.tap.assert_called_once_with(switch['position'])
        self.assertIn(('switch', 16, 8, 5, 0), self.runner.switch_skip)
        self.assertEqual(self.runner.world.count('structure_input', reset_on=('win', 'switch')), 1)

    def test_gate_enemy_is_tapped_and_explore_stays_available(self):
        """探索到守关敌人：点脚圈、按原生寻路等备战；之后同一受阻目标还能再探索，三次后只找机关。"""
        enemy = ring('ring-3', (833., 472.))
        self.runner.wait_entry.return_value = True
        with patch.object(structure, 'recenter_squad'):
            result, _ = self.recover(dict(state='enemy', targets=[enemy], field=self.field, views=4, pans=3))
            self.assertEqual(result, 'entry')
            self.runner.session.tap.assert_called_once_with([833., 472.])
            self.assertEqual(self.runner.wait_entry.call_args.kwargs['target']['click'], [833., 472.])
            self.assertEqual(self.runner.world.count('structure_input', reset_on=('win', 'switch')), 0)
            self.runner.wait_entry.return_value = False
            for _ in range(2):
                result, finder = self.recover(dict(state='enemy', targets=[enemy], field=self.field))
            self.assertEqual(result, 'scouted')
            self.assertEqual(finder.call_args.kwargs['want'], ('switch', 'enemy'))
            self.runner.rejected.add.assert_called()
            result, finder = self.recover(dict(state='exhausted', field=self.field))
        self.assertEqual(finder.call_args.kwargs['want'], ('switch',))

    def test_goal_view_switch_is_tried_before_explorer(self):
        """实录22-21：目标旁的开关先踩，不占探索；原位两帧开启即确认，道路版本号加一。"""
        switch = dict(fixture(position=(1041., 467.)), search_key=('switch', 13, 6, 0, 0))
        look = Mock(return_value=('found', switch, self.field))
        with patch.object(structure, 'recenter_squad') as recenter, \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'activated')) as watch:
            result, finder = self.recover(dict(state='exhausted'), look)
        self.assertEqual(result, 'attempted')
        finder.assert_not_called()
        recenter.assert_not_called()
        self.assertEqual(look.call_args.kwargs, dict(goal=True))
        self.runner.session.tap.assert_called_once_with([1041., 467.])
        watch.assert_called_once()
        # 开关确认开启后受阻记录全部失效（目标旁、探索次数、交互预算），按新道路重新判断。
        self.assertEqual(self.runner.world.history, ['switch'])
        self.assertTrue(self.runner.world.once('goal_view', 'quest:(21, 15)', reset_on=('switch',)))
        self.assertTrue(self.runner.world.once('switch_hunt', 'quest:(21, 15)', reset_on=('switch',)))
        self.assertEqual(self.runner.world.count('structure_input', reset_on=('win', 'switch')), 0)

    def test_unconfirmed_switch_keeps_marks(self):
        """点了开关但原位没确认开启：只算走过设施，同一状态下不重复点同一处、不再看目标旁。"""
        switch = dict(fixture(position=(1041., 467.)), search_key=('switch', 13, 6, 0, 0))
        look = Mock(return_value=('found', switch, self.field))
        with patch.object(structure, 'recenter_squad'), \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'unconfirmed')):
            self.recover(dict(state='exhausted'), look)
        self.assertIn(('switch', 13, 6, 0, 0), self.runner.switch_skip)
        self.assertEqual(self.runner.world.history, ['facility'])
        self.assertFalse(self.runner.world.once('goal_view', 'quest:(21, 15)', reset_on=('switch',)))
        self.assertEqual(self.runner.world.count('structure_input', reset_on=('win', 'switch')), 1)

    def known_catalog(self):
        """目录：远近两个未用过的关闭机关，另有已开启、无底图坐标的机关和一个平台。"""
        catalog = Mock()
        catalog.observe.return_value = []
        catalog.project_facilities.return_value = {}
        catalog.eligible.return_value = True
        catalog.data = dict(survey={}, facilities={
            'scene-1': dict(kind='ground_mechanism', activated='observed_off', map_position=[900., 0.]),
            'scene-2': dict(kind='ground_mechanism', activated='unconfirmed', map_position=[100., 0.]),
            'scene-3': dict(kind='ground_mechanism', activated='observed_on', map_position=[10., 0.]),
            'scene-4': dict(kind='ground_mechanism', activated='observed_off'),
            'scene-5': dict(kind='elevator', activated='unconfirmed', map_position=[5., 0.])})
        return catalog

    def test_known_catalog_switch_is_seen_before_goal_view_and_explorer(self):
        """用户反馈22-36：目录知道开关没用过就直接去试；按底图坐标由近及远，点击仍用新鲜检测。"""
        catalog = self.known_catalog()
        switch = dict(fixture(position=(900., 600.)), search_key=('switch', 11, 8, 0, 0))
        seek = Mock(return_value=dict(client=np.array([890., 610.]), field=self.field))
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'record_map_positions', return_value=None), \
                patch.object(structure, 'locate_view', return_value={}), \
                patch.object(structure, 'gaze_position', return_value=np.zeros(2)), \
                patch.object(structure, 'seek_map_point', seek), \
                patch.object(structure, 'watch_switch', return_value=(self.field, 'unconfirmed')):
            result, finder = self.recover(None, look=Mock(return_value=('found', switch, self.field)))
        self.assertEqual(result, 'attempted')
        np.testing.assert_allclose(seek.call_args.args[1], [100., 0.])
        finder.assert_not_called()
        self.runner.session.tap.assert_called_once_with([900., 600.])
        self.assertEqual(self.runner.world.members('known_switch_tried', 'quest:(21, 15)', reset_on=('switch',)),
                         {'scene-2'})

    def test_known_switch_misses_recenter_then_fall_back_to_explorer(self):
        """已知机关处都没新鲜确认到：每个只看一次，已开启和无坐标的不看，找回小队后照常探索。"""
        catalog = self.known_catalog()
        seek = Mock(return_value=dict(client=np.array([890., 610.]), field=self.field))
        with patch.object(structure, 'facility_catalog', return_value=catalog), \
                patch.object(structure, 'record_map_positions', return_value=None), \
                patch.object(structure, 'locate_view', return_value={}), \
                patch.object(structure, 'gaze_position', return_value=np.zeros(2)), \
                patch.object(structure, 'seek_map_point', seek), \
                patch.object(structure, 'recenter_squad') as recenter:
            result, finder = self.recover(dict(state='exhausted', field=self.field))
        self.assertEqual(result, 'scouted')
        self.assertEqual([call.args[1].tolist() for call in seek.call_args_list], [[100., 0.], [900., 0.]])
        recenter.assert_called()
        finder.return_value.run.assert_called_once()
        self.assertEqual(self.runner.world.members('known_switch_tried', 'quest:(21, 15)', reset_on=('switch',)),
                         {'scene-1', 'scene-2'})

    def test_recenter_skipped_when_squad_already_visible(self):
        """实录22-27：任务定位后小队仍在画面内，没有离屏头像可点，不算找回失败。"""
        with patch.object(structure, 'squad_arrow', return_value=np.array([885., 420.])), \
                patch('module.campaign.vision.minimap_occlusion.recenter_offscreen') as offscreen:
            self.assertIsNone(structure.recenter_squad(self.runner))
        offscreen.assert_not_called()

    def test_empty_goal_view_recenters_squad_then_explores_once(self):
        """目标旁没有开关：回到小队再探索；同一道路状态下一次受阻不再看目标旁、不再探索。"""
        look = Mock(return_value=('none', None, self.field))
        with patch.object(structure, 'recenter_squad') as recenter:
            result, finder = self.recover(dict(state='exhausted', field=self.field), look)
            self.assertEqual(result, 'scouted')
            finder.return_value.run.assert_called_once()
            self.assertEqual(recenter.call_count, 2)
            look.reset_mock()
            result, finder = self.recover(dict(state='exhausted', field=self.field), look)
        look.assert_not_called()
        finder.assert_not_called()

    def test_switch_watch_confirms_on_state_or_ends_when_switch_leaves_view(self):
        """点开关后盯原位：连续两帧开启才确认；原位连续三帧不见开关即结束观察。"""
        target = fixture(position=(1041., 467.))
        frames = iter([[target], [dict(target, state='on')], [dict(target, state='on', position=[1043., 468.])]])
        with patch.object(structure, 'detect_objects', side_effect=lambda *a: next(frames)), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.):
            self.assertEqual(structure.watch_switch(self.runner, target)[1], 'activated')
            frames = iter([[target], [], [], []])
            self.assertEqual(structure.watch_switch(self.runner, target)[1], 'unconfirmed')

    def test_explore_budget_fits_blocked_goal_limit(self):
        """受阻探索时限须小于同一受阻目标的总时限，留出踩机关和重试的时间。"""
        self.assertLess(structure.EXPLORE_SECONDS, structure.BLOCKED_GOAL_SECONDS)

    def test_blocked_goal_past_time_limit_stops_and_reports(self):
        """同一受阻目标超过总时限即停止，不再搜索或点击。"""
        self.runner.world.put('blocked_since', 'quest:(21, 15)', 0.)
        with patch.object(structure.time, 'monotonic', return_value=structure.BLOCKED_GOAL_SECONDS+1.):
            with self.assertRaisesRegex(structure.StructureUnconfirmed, '分钟'):
                self.recover(dict(state='exhausted'))
        self.runner.session.tap.assert_not_called()

    def test_exhausted_explore_rescans_then_falls_back_to_platform(self):
        """探索无果不点旧帧平台；下一次受阻不再探索，改为新鲜确认的平台；预算用完才停止。"""
        with patch.object(structure, 'recenter_squad'):
            result, _ = self.recover(dict(state='exhausted', views=21, pans=20, field=self.field))
            self.assertEqual(result, 'scouted')
            self.runner.session.tap.assert_not_called()
            result, finder = self.recover(dict(state='exhausted'))
            self.assertEqual(result, 'attempted')
            finder.assert_not_called()
            self.runner.session.tap.assert_called_once_with([726., 605.])
            self.recover(dict(state='exhausted'))
            with self.assertRaises(structure.StructureUnconfirmed):
                self.recover(dict(state='exhausted'))

    def test_missing_squad_anchor_uses_view_center_instead_of_stopping(self):
        """停止策略分级：小队箭头和小地图圆环都读不到时按画面中心排序，照常找机关，不整轮停止。"""
        pad = fixture('elevator', 'end', (726., 605.))
        finder = Mock()
        finder.return_value.run.return_value = dict(state='exhausted', field=self.field)
        with patch.object(structure, 'detect_objects', return_value=[pad]), \
                patch.object(structure, 'squad_arrow', return_value=None), \
                patch.object(structure.goto, 'battle_popup_score', return_value=0.), \
                patch.object(explore, 'Explorer', finder), \
                patch.object(structure, 'switch_look', Mock(return_value=('none', None, self.field))), \
                patch.object(structure, 'recenter_squad'):
            self.assertEqual(structure.recover_visible_structure(self.runner, self.field), 'scouted')
        finder.return_value.run.assert_called_once()


class WorldModelTests(unittest.TestCase):
    """道路事件让对应记录自动失效，替代各处手动清空的标记。"""

    def test_marks_expire_on_their_reset_events(self):
        world = WorldModel()
        self.assertTrue(world.once('goal_view', 'g', reset_on=('switch',)))
        self.assertFalse(world.once('goal_view', 'g', reset_on=('switch',)))
        self.assertEqual(world.bump('structure_input', reset_on=('win', 'switch')), 1)
        self.assertEqual(world.bump('marker_pan'), 1)
        world.advance('facility')
        self.assertFalse(world.once('goal_view', 'g', reset_on=('switch',)))
        self.assertEqual(world.count('structure_input', reset_on=('win', 'switch')), 1)
        self.assertEqual(world.count('marker_pan'), 0)
        world.advance('switch')
        self.assertTrue(world.once('goal_view', 'g', reset_on=('switch',)))
        self.assertEqual(world.bump('structure_input', reset_on=('win', 'switch')), 1)

    def test_never_marks_and_members(self):
        world = WorldModel()
        world.put('marker_walk', (20, 5), True)
        world.add_member('tried', 'g', 'a', reset_on=('switch',))
        world.add_member('tried', 'g', 'b', reset_on=('switch',))
        world.advance('win')
        self.assertTrue(world.get('marker_walk', (20, 5), reset_on=()))
        self.assertEqual(world.members('tried', 'g', reset_on=('switch',)), {'a', 'b'})
        world.advance('switch')
        self.assertEqual(world.members('tried', 'g', reset_on=('switch',)), set())
        with self.assertRaises(ValueError):
            world.advance('teleport')

    def test_listeners_run_on_every_event(self):
        world = WorldModel()
        seen = []
        world.listeners.append(seen.append)
        world.advance('win')
        world.advance('switch')
        self.assertEqual(seen, ['win', 'switch'])


if __name__ == '__main__':
    unittest.main()
