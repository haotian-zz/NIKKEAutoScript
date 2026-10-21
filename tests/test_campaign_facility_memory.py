"""以已知镜头变换验证跨帧身份和持久尝试；不连接窗口，不加载检测模型。"""

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign.facility_memory import FacilityMemory, FacilityUnconfirmed


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

    def test_neighboring_facilities_keep_distinct_histories(self):
        """两个近邻设施平移后各自关联，尝试其中一个不封禁另一个。"""
        first = self.catalog.observe(self.field, [observed(), observed((900., 500.))])
        self.catalog.attempt(first[0]['facility_id'], 'blocked-A')
        after = self.catalog.observe(self.panned, [observed((980., 500.)), observed((1080., 500.))])
        self.assertEqual([item['facility_id'] for item in first], [item['facility_id'] for item in after])
        self.assertTrue(self.catalog.eligible(after[1]['facility_id'], 'blocked-A'))

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


if __name__ == '__main__':
    unittest.main()
