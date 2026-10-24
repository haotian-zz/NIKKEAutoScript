"""章节支线委托：「!」菱形、SUB 标题、对话列表行、「直接前往」与地图锚点识别。"""

import unittest

import cv2
import numpy as np

from module.side_quest import side_quest

ORANGE = (30, 150, 250)
WHITE = (255, 255, 255)
YELLOW = (50, 190, 250)


def field():
    image = np.random.RandomState(0).randint(30, 70, (1280, 720, 3)).astype(np.uint8)
    return image


def add_marker(image, x, y, size=82):
    """橙框白底菱形，中间黑色「!」。"""
    half = size // 2
    outer = np.array([[x, y - half], [x + half, y], [x, y + half], [x - half, y]])
    inner = np.array([[x, y - half + 10], [x + half - 10, y], [x, y + half - 10], [x - half + 10, y]])
    cv2.fillPoly(image, [outer], ORANGE)
    cv2.fillPoly(image, [inner], WHITE)
    image[y - 20:y + 6, x - 3:x + 3] = 0
    return image


class MarkerTests(unittest.TestCase):
    def test_markers_sorted_by_distance_to_center(self):
        image = add_marker(add_marker(field(), 120, 900), 400, 600)
        markers = side_quest.quest_markers(image)
        self.assertEqual(len(markers), 2)
        self.assertTrue(abs(markers[0][0] - 400) <= 2 and abs(markers[0][1] - 600) <= 2)

    def test_edge_marker_cut_in_half_still_found(self):
        """实录：屏幕左缘只露出一半的菱形宽约 59px，也是支线目标。"""
        image = add_marker(field(), 18, 560)
        self.assertEqual(len(side_quest.quest_markers(image)), 1)

    def test_orange_without_white_center_ignored(self):
        image = field()
        image[600:680, 300:380] = ORANGE
        self.assertEqual(side_quest.quest_markers(image), [])

    def test_marker_under_ui_ignored(self):
        """小地图、任务栏、底部按钮和顶栏上的菱形不点（要先拖镜头）。"""
        image = add_marker(add_marker(add_marker(field(), 150, 250), 540, 1200), 590, 25)
        self.assertEqual(side_quest.quest_markers(image), [])


class TitleTests(unittest.TestCase):
    def test_sub_stage(self):
        self.assertEqual(side_quest.sub_stage('SUB 18-8 STAGE'), (18, 8))
        self.assertEqual(side_quest.sub_stage('SUB18－7STAGE'), (18, 7))

    def test_other_titles_rejected(self):
        for text in ('18-8 STAGE', 'EX STAGE', 'HARD 4-5 STAGE', ''):
            self.assertIsNone(side_quest.sub_stage(text))


class ListTests(unittest.TestCase):
    def test_rows_from_avatars(self):
        image = np.full((1280, 720, 3), 245, np.uint8)
        for y in (432, 539, 646):
            image[y - 45:y + 45, 70:180] = YELLOW
        image[1120:1160, 70:180] = YELLOW  # 只露出一截的行不算
        rows = side_quest.chat_rows(image)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(abs(a - b) <= 2 for a, b in zip(rows, (432, 539, 646))))

    def test_seen_row_identity(self):
        image = np.full((1280, 720, 3), 245, np.uint8)
        image[400:420, 230:300] = 40
        image[507:527, 230:340] = 40
        a, b = side_quest.row_title(image, 432), side_quest.row_title(image, 539)
        self.assertTrue(side_quest.same_row(a, side_quest.row_title(image.copy(), 432)))
        self.assertFalse(side_quest.same_row(a, b))


class TemplateTests(unittest.TestCase):
    def test_quest_go_button(self):
        image = np.full((1280, 720, 3), 245, np.uint8)
        h, w = side_quest.QUEST_GO.shape[:2]
        image[405:405 + h, 490:490 + w] = side_quest.QUEST_GO
        x, y = side_quest.quest_go(image)
        self.assertTrue(abs(x - (490 + w // 2)) <= 1 and abs(y - (405 + h // 2)) <= 1)
        self.assertIsNone(side_quest.quest_go(np.full((1280, 720, 3), 245, np.uint8)))

    def test_quest_done_button(self):
        image = np.full((1280, 720, 3), 245, np.uint8)
        h, w = side_quest.QUEST_DONE.shape[:2]
        image[405:405 + h, 490:490 + w] = side_quest.QUEST_DONE
        self.assertIsNotNone(side_quest.quest_done(image))
        self.assertIsNone(side_quest.quest_go(image))

    def test_quest_talk_button(self):
        """实录：「寻找伏特Jr.」弹的是蓝色「对话」按钮，不是进入战斗。"""
        image = field()
        self.assertIsNone(side_quest.quest_talk(image))
        h, w = side_quest.QUEST_TALK.shape[:2]
        image[1160:1160 + h, 300:300 + w] = side_quest.QUEST_TALK
        x, y = side_quest.quest_talk(image)
        self.assertTrue(abs(x - side_quest.QUEST_TALK_POINT[0]) <= 5 and abs(y - side_quest.QUEST_TALK_POINT[1]) <= 5)

    def test_map_anchor(self):
        image = field()
        self.assertFalse(side_quest.map_visible(image))
        h, w = side_quest.MAP_CHECK.shape[:2]
        image[1216:1216 + h, 492:492 + w] = side_quest.MAP_CHECK
        self.assertTrue(side_quest.map_visible(image))


if __name__ == '__main__':
    unittest.main()
