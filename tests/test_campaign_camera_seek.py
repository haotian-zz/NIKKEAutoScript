"""外部观察区域不能越过身份、地图及范围约束。"""

import unittest

from module.campaign.navigation.camera_seek import validate_hint


class CameraHintTests(unittest.TestCase):
    """只验证研究线索入口；不连接游戏，不将线索当作可点击设施。"""

    def test_wrong_map_identity_or_action_purpose_rejected(self):
        """外部线索不能跨章或底图复用，也不能被混成移动授权。"""
        hint = dict(purpose='camera_region_only', identity=[21, 'normal'], map_sha256='map-a',
                    center_map=[958., 2148.], radius_map=180.)
        for patch in (dict(map_sha256='map-b'), dict(identity=[22, 'normal']), dict(purpose='click_target')):
            with self.assertRaises(ValueError):
                validate_hint(dict(hint, **patch), (21, 'normal'), 'map-a', (1958, 2251))
        point, radius = validate_hint(hint, (21, 'normal'), 'map-a', (1958, 2251))
        self.assertEqual(point.tolist(), [958., 2148.])
        self.assertEqual(radius, 180.)

    def test_nonfinite_outside_and_unbounded_radius_rejected(self):
        """非有限或地图外位置，以及无限扩大观察范围均拒绝。"""
        hint = dict(purpose='camera_region_only', identity=[21, 'normal'], map_sha256='map-a',
                    center_map=[100., 200.], radius_map=180.)
        for patch in (dict(center_map=[float('nan'), 1.]), dict(center_map=[2000., 2.]), dict(radius_map=1000.)):
            with self.assertRaises(ValueError):
                validate_hint(dict(hint, **patch), (21, 'normal'), 'map-a', (1958, 2251))


if __name__ == '__main__':
    unittest.main()
