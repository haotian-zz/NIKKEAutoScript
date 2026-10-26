"""以已知镜头变换验证跨帧身份和持久尝试；不连接窗口，不加载检测模型。"""

import tempfile
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign.facility_memory import FacilityMemory, FacilityUnconfirmed, facility_frame, voted_facilities


def facility_frame_for(field):
    """无设施时的关键帧特征。"""
    return facility_frame(field, [])


def observed(position=(800., 500.), state='start', kind='elevator'):
    """构造已通过调用方双帧确认的检测，身份关联须由真实背景特征提供。"""
    x, y = position
    return dict(kind=kind, state=state, confidence=.95, position=[x, y], box=[x-30, y-20, x+30, y+20])


class FacilityMemoryTests(unittest.TestCase):
    """模拟相邻镜头、程序重启及纹理丢失，保证不能靠换编号绕过已尝试记录。"""

    @classmethod
    def setUpClass(cls):
        """固定随机纹理产生可复现特征，平移180px模拟有重叠的真实相机拖动。"""
        rng = np.random.default_rng(791)
        small = rng.integers(0, 256, (500, 888), dtype=np.uint8)
        gray = cv2.resize(small, (1776, 999))
        cls.field = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cls.panned = cv2.warpAffine(cls.field, np.float32([[1, 0, 180], [0, 1, 0]]), (1776, 999))

    def setUp(self):
        """每项验证独立磁盘目录，不读写真实实例的机关状态。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.catalog = FacilityMemory('isolated', (21, 'normal'), 'sha-A', root=self.temporary.name)

    def test_pan_and_visual_role_change_preserve_attempt_after_restart(self):
        """start变end并平移镜头、重启后仍是原设施，同受阻目标不能重复尝试。"""
        identifier = self.catalog.observe(self.field, [observed()])[0]['facility_id']
        self.catalog.attempt(identifier, 'quest:21-15')
        restarted = FacilityMemory('isolated', (21, 'normal'), 'sha-A', root=self.temporary.name)
        item = restarted.observe(self.panned, [observed((980., 500.), 'end')])[0]
        self.assertEqual(item['facility_id'], identifier)
        self.assertFalse(restarted.eligible(identifier, 'quest:21-15'))
        self.assertTrue(restarted.eligible(identifier, 'quest:21-16'))
        self.assertEqual(restarted.data['connections'], [])

    def test_transit_without_continuation_does_not_count(self):
        """中途换乘又没能续点的尝试不算试过（重启后同样），最多三次；续点过的中途换乘算试过。"""
        identifier = self.catalog.observe(self.field, [observed()])[0]['facility_id']
        self.catalog.attempt(identifier, 'quest:None')
        self.assertFalse(self.catalog.eligible(identifier, 'quest:None'))
        self.catalog.transit(identifier, continued=False)
        restarted = FacilityMemory('isolated', (21, 'normal'), 'sha-A', root=self.temporary.name)
        self.assertTrue(restarted.eligible(identifier, 'quest:None'))
        for _ in range(2):
            restarted.attempt(identifier, 'quest:None')
            restarted.transit(identifier, continued=False)
        self.assertFalse(restarted.eligible(identifier, 'quest:None'))
        other = self.catalog.observe(self.field, [observed(), observed((1100., 500.))])[1]['facility_id']
        self.catalog.attempt(other, 'quest:None')
        self.catalog.transit(other, continued=True)
        self.assertFalse(self.catalog.eligible(other, 'quest:None'))

    def test_project_facility_follows_camera_pan(self):
        """镜头平移180px后设施投影跟着平移；无纹理画面配不上，返回None。"""
        identifier = self.catalog.observe(self.field, [observed()])[0]['facility_id']
        np.testing.assert_allclose(self.catalog.project_facility(self.panned, identifier), [980., 500.], atol=2.)
        self.assertIsNone(self.catalog.project_facility(np.zeros_like(self.field), identifier))

    def test_click_and_motion_do_not_activate_switch(self):
        """点击及192px地图位移不能认领机关开启，只有重新观察on状态才记录observed_on。"""
        identifier = self.catalog.observe(
            self.field, [observed(kind='ground_mechanism', state='off')])[0]['facility_id']
        self.catalog.attempt(identifier, 'blocked-A')
        self.catalog.feedback(identifier, dict(state='moved', distance_map_pixels=192.))
        self.assertEqual(self.catalog.data['facilities'][identifier]['activated'], 'observed_off')
        self.catalog.observe(self.panned, [observed((980., 500.), 'on', 'ground_mechanism')])
        self.assertEqual(self.catalog.data['facilities'][identifier]['activated'], 'observed_on')
        self.assertFalse(self.catalog.eligible(identifier, 'blocked-B'))
        self.assertEqual(self.catalog.data['facilities'][identifier]['attempts'][0]['gate'], 'unconfirmed')

    def test_verified_switch_change_allows_new_transfer_attempt(self):
        """道路前置可能变化时，只在同一机关off到on被实测确认后开放下一轮尝试。"""
        items = self.catalog.observe(self.field, [observed(), observed((1100., 500.), 'off', 'ground_mechanism')])
        platform = items[0]['facility_id']
        self.catalog.attempt(platform, 'blocked-A')
        self.assertFalse(self.catalog.eligible(platform, 'blocked-A'))
        self.catalog.observe(self.field, [observed(), observed((1100., 500.), 'on', 'ground_mechanism')])
        self.assertTrue(self.catalog.eligible(platform, 'blocked-A'))

    def test_lost_registration_cannot_create_fresh_identity(self):
        """不相关或无纹理画面必须报未确认，不能给同一已试设施重新编号。"""
        self.catalog.observe(self.field, [observed()])
        with self.assertRaises(FacilityUnconfirmed):
            self.catalog.observe(np.zeros_like(self.field), [observed()])
        self.assertEqual(len(self.catalog.data['facilities']), 1)

    def test_background_bridge_keeps_attempts_across_nonoverlapping_end_views(self):
        """两端无重叠时不能直接关联，中间关键帧须恢复连续性且不改已有尝试及开关状态。"""
        rng = np.random.default_rng(901)
        gray = cv2.resize(rng.integers(0, 256, (500, 2000), dtype=np.uint8), (4000, 999))
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        first, middle, last = [canvas[:, x:x+1776] for x in (0, 900, 1800)]
        identifier = self.catalog.observe(first, [observed()])[0]['facility_id']
        self.catalog.attempt(identifier, 'blocked-A')
        before = copy.deepcopy(self.catalog.data['facilities'])
        self.assertFalse(self.catalog.track_background(last, force=True))
        self.assertTrue(self.catalog.track_background(middle, force=True))
        self.assertTrue(self.catalog.track_background(last, force=True))
        self.assertEqual(self.catalog.data['facilities'], before)
        self.assertIn(identifier, self.catalog.current)
        restarted = FacilityMemory('isolated', (21, 'normal'), 'sha-A', root=self.temporary.name)
        self.assertTrue(restarted.track_background(last, force=True))
        self.assertFalse(restarted.eligible(identifier, 'blocked-A'))

    def test_background_on_newest_screen_stops_after_one_registration(self):
        """实测22章46个关键帧逐个配准约1.5秒：轮询同一屏时命中最新关键帧即停，不新增关键帧。"""
        self.catalog.observe(self.panned, [])
        self.catalog.observe(self.field, [])
        views = len(self.catalog.data['views'])
        names = FacilityMemory.observe.__globals__
        register = Mock(wraps=names['scene_registration'])
        with patch.dict(names, scene_registration=register), patch.object(self.catalog, 'save') as save:
            self.assertTrue(self.catalog.track_background(self.field, force=True))
        self.assertEqual(register.call_count, 1)
        save.assert_not_called()
        self.assertEqual(len(self.catalog.data['views']), views)

    def test_known_off_switch_is_projected_as_hint_until_seen_again(self):
        """历史关闭机关在新视野中被挡住时，按配准投影出应在位置作线索；本帧检测到时不重复列出。"""
        switch = observed((800., 500.), 'off', 'ground_mechanism')
        identifier = self.catalog.observe(self.field, [switch])[0]['facility_id']
        self.catalog.observe(self.panned, [])
        self.assertEqual([hint['facility_id'] for hint in self.catalog.expected], [identifier])
        np.testing.assert_allclose(self.catalog.expected[0]['position'], [980., 500.], atol=3.)
        self.catalog.observe(self.panned, [observed((980., 500.), 'off', 'ground_mechanism')])
        self.assertEqual(self.catalog.expected, [])

    def test_second_instance_reuses_shared_positions_without_attempts(self):
        """另一个实例首次进入同章同底图时复用设施位置与关键帧，尝试与开关状态从未确认开始。"""
        identifier = self.catalog.observe(
            self.field, [observed(kind='ground_mechanism', state='off')])[0]['facility_id']
        self.catalog.attempt(identifier, 'quest:(21, 32)')
        other = FacilityMemory('another-account', (21, 'normal'), 'sha-A', root=self.temporary.name)
        record = other.data['facilities'][identifier]
        self.assertEqual(record['attempts'], [])
        self.assertEqual(record['activated'], 'unconfirmed')
        self.assertTrue(other.eligible(identifier, 'quest:(21, 32)'))
        item = other.observe(self.panned, [observed((980., 500.), 'off', 'ground_mechanism')])[0]
        self.assertEqual(item['facility_id'], identifier)
        isolated = FacilityMemory('another-account', (21, 'normal'), 'sha-B', root=self.temporary.name)
        self.assertEqual(isolated.data['facilities'], {})

    def test_votes_survive_one_smoke_miss_but_reject_flashes_and_low_scores(self):
        """实录21-32：烟雾三帧漏一帧仍接受；单帧闪现、低分误检（坦克舱盖约0.6）不接受。"""
        switch = observed((960., 492.), 'off', 'ground_mechanism')
        hatch = dict(observed((567., 258.), 'off', 'ground_mechanism'), confidence=.6)
        accepted = voted_facilities([[], [switch, hatch], [dict(switch, position=[961., 493.])], [hatch]])
        self.assertEqual([item['position'] for item in accepted], [[961., 493.]])
        self.assertEqual(voted_facilities([[switch], [], [], []]), [])

    def test_neighboring_facilities_keep_distinct_histories(self):
        """两个近邻设施平移后各自关联，尝试其中一个不封禁另一个。"""
        first = self.catalog.observe(self.field, [observed(), observed((900., 500.))])
        self.catalog.attempt(first[0]['facility_id'], 'blocked-A')
        after = self.catalog.observe(self.panned, [observed((980., 500.)), observed((1080., 500.))])
        self.assertEqual([item['facility_id'] for item in first], [item['facility_id'] for item in after])
        self.assertTrue(self.catalog.eligible(after[1]['facility_id'], 'blocked-A'))

    def test_duplicate_history_skips_only_that_facility(self):
        """实录 2026-10-25（HARD 9）：同一机关登记了两次，身份不明时只跳过它，不整轮停止。"""
        self.catalog.observe(self.field, [observed(kind='ground_mechanism', state='off'),
                                          observed((805., 500.), kind='ground_mechanism', state='off'),
                                          observed((800., 800.))])
        after = self.catalog.observe(self.panned, [observed((982., 500.), kind='ground_mechanism', state='off'),
                                                   observed((980., 800.))])
        self.assertEqual([item['kind'] for item in after], ['elevator'])
        self.assertEqual(self.catalog.unresolved[0]['reason'], 'ambiguous_identity')

    def test_resource_coordinates_and_scope_are_not_scene_clicks(self):
        """外部有向传送标注保留方向且不冒充已激活；其他底图记录隔离。"""
        document = dict(image_sha256='sha-A', objects=[dict(id=name, type='point',
                        category='ground_elevator', points=[[x, 200.]]) for name, x in [('a', 100.), ('b', 400.)]],
                        connections=[dict(category='elevator_connection', **{'from': 'a', 'to': 'b'}, directed=True)])
        self.catalog.import_annotations(document)
        self.assertEqual(len(self.catalog.data['connections']), 1)
        self.assertTrue(self.catalog.data['connections'][0]['directed'])
        self.assertEqual(self.catalog.current, {})
        self.assertEqual(self.catalog.data['facilities']['map-a']['activated'], 'unconfirmed')
        other = FacilityMemory('isolated', (21, 'normal'), 'sha-B', root=self.temporary.name)
        self.assertEqual(other.data['facilities'], {})
        with self.assertRaises(FacilityUnconfirmed):
            other.import_annotations(document)

    def test_full_catalog_evicts_oldest_bridge_but_never_anchor_views(self):
        """实录22-36：64个关键帧40个是桥接帧，满了巡检一屏都没登记；满了只淘汰最久未命中的桥接帧。"""
        rng = np.random.default_rng(902)
        gray = cv2.resize(rng.integers(0, 256, (500, 2400), dtype=np.uint8), (4800, 999))
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        strips = [canvas[:, x:x+1776] for x in (0, 600, 1200, 1800)]
        names = FacilityMemory.observe.__globals__
        with patch.dict(names, MAX_VIEWS=3):
            identifier = self.catalog.observe(strips[0], [observed()])[0]['facility_id']
            for strip in strips[1:]:
                self.assertTrue(self.catalog.track_background(strip, force=True))
            self.assertEqual(len(self.catalog.data['views']), 3)
            self.assertEqual(self.catalog.anchors(), {0})
            self.assertIn(0, self.catalog.frames)
            self.assertEqual(len(list(self.catalog.folder.glob('view-*.npz'))), 3)
            restarted = FacilityMemory('isolated', (21, 'normal'), 'sha-A', root=self.temporary.name)
            self.assertEqual(restarted.observe(strips[0], [observed()])[0]['facility_id'], identifier)
            only_anchor = FacilityMemory('anchors', (22, 'normal'), 'sha-A', root=self.temporary.name)
        with patch.dict(names, MAX_VIEWS=1):
            only_anchor.observe(strips[0], [observed()])
            with self.assertRaises(FacilityUnconfirmed):
                only_anchor.observe(strips[1], [observed((300., 500.), 'off', 'ground_mechanism')])

    def test_old_observations_stop_anchoring_views(self):
        """每个设施只保留最近几次视野观测，更早的视野变回可淘汰的桥接帧。"""
        names = FacilityMemory.observe.__globals__
        with patch.dict(names, MAX_OBSERVATIONS=1):
            identifier = self.catalog.observe(self.field, [observed()])[0]['facility_id']
            self.catalog.observe(self.panned, [observed((980., 500.))])
        self.assertEqual([old['view'] for old in self.catalog.data['facilities'][identifier]['observations']], [1])
        self.assertEqual(self.catalog.anchors(), {1})

    def test_map_positions_average_estimates_and_keep_annotations(self):
        """实测底图坐标取平均；标注坐标不改写，只报告偏差；共享副本带坐标。"""
        items = self.catalog.observe(self.field, [observed(kind='ground_mechanism', state='off')])
        identifier = items[0]['facility_id']
        self.assertEqual(self.catalog.needs_map(items), items)
        self.assertEqual(self.catalog.place(items, lambda point: np.asarray(point)/10.), [])
        errors = self.catalog.place(items, lambda point: np.asarray(point)/10.+[2., 0.])
        record = self.catalog.data['facilities'][identifier]
        self.assertEqual(errors, [dict(facility_id=identifier, source='viewport_estimate', error=2.)])
        np.testing.assert_allclose(record['map_position'], [81., 50.])
        self.assertEqual(record['map_samples'], 2)
        self.assertEqual(self.catalog.needs_map(items), [])
        self.catalog._new('map-a', 'ground_mechanism', 'annotations')
        self.catalog.data['facilities']['map-a']['map_position'] = [10., 10.]
        errors = self.catalog.place([dict(facility_id='map-a', position=[0., 0.])], lambda point: [13., 14.])
        self.assertEqual(errors[0]['error'], 5.)
        self.assertEqual(self.catalog.data['facilities']['map-a']['map_position'], [10., 10.])
        other = FacilityMemory('another-account', (21, 'normal'), 'sha-A', root=self.temporary.name)
        np.testing.assert_allclose(other.data['facilities'][identifier]['map_position'], [81., 50.])
        self.assertEqual(other.data['facilities'][identifier]['map_source'], 'viewport_estimate')

    def test_feature_votes_pick_overlapping_keyframes_before_full_scan(self):
        """实测22章64个关键帧逐个配准约3秒：先按特征投票只配准票数最高的几个，配上即不再全查。"""
        rng = np.random.default_rng(903)
        gray = cv2.resize(rng.integers(0, 256, (500, 3200), dtype=np.uint8), (6400, 999))
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        names = FacilityMemory.observe.__globals__
        for x in (0, 1200, 2400, 3600):
            self.catalog.observe(canvas[:, x:x+1776], [])
        register = Mock(wraps=names['scene_registration'])
        with patch.dict(names, scene_registration=register, CANDIDATE_VIEWS=1):
            self.catalog.observe(canvas[:, 1300:1300+1776], [])
        self.assertEqual(register.call_count, 1)
        self.assertIs(register.call_args_list[0].args[0], self.catalog.frames[1])

    def test_manual_entry_is_projected_after_camera_moves(self):
        """实录22-36：EX 关闭后换镜头就是新编号；按关键帧记下位置，平移后的画面能投影出它。"""
        self.assertFalse(self.catalog.has_manual_entries())
        registered = self.catalog.remember_manual(self.field, [800., 500.], (22, 0, 'EX', 1))
        identifier = registered[0]['facility_id']
        self.assertEqual(self.catalog.data['facilities'][identifier]['stage'], [22, 0, 'EX', 1])
        self.assertTrue(self.catalog.has_manual_entries())
        self.catalog.observe(self.panned, [])
        self.assertEqual([entry['facility_id'] for entry in self.catalog.manual_projected], [identifier])
        np.testing.assert_allclose(self.catalog.manual_projected[0]['position'], [980., 500.], atol=3.)
        again = self.catalog.remember_manual(self.panned, [985., 505.], (22, 0, 'EX', 1))
        self.assertEqual(again[0]['facility_id'], identifier)

    def test_vote_candidates_always_include_anchor_views(self):
        """候选全是桥接帧时也带上票数最高的锚点帧，已登记设施仍能投影。"""
        rng = np.random.default_rng(904)
        gray = cv2.resize(rng.integers(0, 256, (500, 2400), dtype=np.uint8), (4800, 999))
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        identifier = self.catalog.remember_manual(canvas[:, 0:1776], [900., 500.], (22, 0, 'EX', 1))[0]['facility_id']
        for x in (100, 200, 300):
            self.catalog.observe(canvas[:, x:x+1776], [])
        names = FacilityMemory.observe.__globals__
        with patch.dict(names, CANDIDATE_VIEWS=1, ANCHOR_CANDIDATES=1):
            first, _ = self.catalog._vote_order(facility_frame_for(canvas[:, 350:350+1776]))
            self.assertIn(0, first)
            self.catalog.observe(canvas[:, 350:350+1776], [])
        self.assertEqual([entry['facility_id'] for entry in self.catalog.manual_projected], [identifier])


if __name__ == '__main__':
    unittest.main()
