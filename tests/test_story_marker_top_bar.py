"""顶栏里的任务菱形只在显式要求时返回、且标为 hud；用 25-18 实录截图的右上角，不加载模型。"""
from pathlib import Path
import unittest

import cv2
import numpy as np

from module.campaign.vision.story_marker import story_markers


def frame():
    canvas = np.zeros((999, 1776, 3), np.uint8)
    canvas[0:260, 1376:1776] = cv2.imread(str(Path(__file__).parent/'fixtures'/'top_bar_marker_25_18.png'))
    return canvas


class TopBarMarkerTests(unittest.TestCase):
    def test_default_keeps_top_bar_masked(self):
        self.assertEqual(story_markers(frame()), [])

    def test_top_bar_marker_is_direction_only(self):
        found = story_markers(frame(), top_bar=True)
        self.assertEqual([item['kind'] for item in found], ['hud'])
        np.testing.assert_allclose(found[0]['position'], [1617., 53.], atol=4.)


if __name__ == '__main__':
    unittest.main()
