"""小队局部平台补查的坐标与边界契约；推理替身不加载权重或输入驱动。"""

import unittest
from unittest.mock import Mock, patch

import numpy as np

from module.campaign.vision.detection import Detection
from module.campaign.vision.objects import nearby_platforms, visible_platforms


class NearbyPlatformTests(unittest.TestCase):
    """边缘小队仍裁取完整400px，不把局部模型坐标直接发送到客户区。"""

    def test_border_crop_keeps_native_size_and_restores_client_coordinates(self):
        """右下边缘只夹裁图起点，输出必须补回实际偏移；无定位不运行模型。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        detector = Mock()
        detector.predict.return_value = [Detection('scene_elevator_start', .95, (60., 70., 160., 170.))]
        with patch('module.campaign.vision.objects.detector', return_value=detector):
            found = nearby_platforms(field, [1750., 960.])
            self.assertEqual(detector.predict.call_args.args[0].shape, (400, 400, 3))
            self.assertEqual(found[0]['box'], [1436., 669., 1536., 769.])
            self.assertEqual(found[0]['view'], 'squad_local_400')
            detector.reset_mock()
            with self.assertRaises(ValueError):
                nearby_platforms(field, [float('nan'), 400.])
            detector.predict.assert_not_called()

    def test_recenter_never_promotes_a_different_platform(self):
        """全屏弱提案补查遇到另一个高分平台时，不能把其分数移植给原目标。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        detector = Mock()
        weak = Detection('scene_elevator_start', .88, (260., 300., 350., 380.))
        different = Detection('scene_elevator_start', .99, (300., 200., 380., 280.))
        detector.predict.side_effect = [[weak]] + [[] for _ in range(14)] + [[different]]
        with patch('module.campaign.vision.objects.detector', return_value=detector):
            found = visible_platforms(field)
        self.assertEqual(detector.predict.call_count, 16)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['confidence'], .88)
        self.assertEqual(found[0]['box'], [260., 300., 350., 380.])


if __name__ == '__main__':
    unittest.main()
