"""竖屏章节列表困难页：列表锚点、困难页签与已解锁卡片识别。"""

import unittest

import numpy as np

from module.campaign.vision import hard_list


def hard_page():
    image = np.full((1280, 720, 3), (30, 34, 38), np.uint8)
    h, w = hard_list.LIST_TAB.shape[:2]
    image[1100:1100 + h, 52:52 + w] = hard_list.LIST_TAB
    image[1103:1145, 510:660] = (55, 58, 156)
    return image


def add_card(image, x1, top, locked=False):
    """卡片：上下 2px 边框，中间是章节图（解锁偏亮，上锁很暗）。"""
    border = 200 if locked else 235
    image[top:top + 2, x1:x1 + 190] = border
    image[top + 40:top + 42, x1:x1 + 190] = border
    image[top + 2:top + 40, x1:x1 + 190] = (40, 40, 45) if locked else (100, 90, 110)
    return image


class HardListTests(unittest.TestCase):
    def test_list_and_tab(self):
        image = hard_page()
        self.assertTrue(hard_list.list_visible(image))
        self.assertTrue(hard_list.hard_tab_active(image))
        image[1103:1145, 510:660] = 240
        self.assertFalse(hard_list.hard_tab_active(image))
        self.assertFalse(hard_list.list_visible(np.zeros((1280, 720, 3), np.uint8)))

    def test_unlocked_cards_only(self):
        """实录排布：第 3 章右列 y≈280、第 4 章左列 y≈400 已解锁，第 5、6 章上锁。"""
        image = hard_page()
        add_card(image, 462, 260)
        add_card(image, 68, 380)
        add_card(image, 462, 500, locked=True)
        add_card(image, 68, 620, locked=True)
        cards = hard_list.unlocked_cards(image)
        self.assertEqual(len(cards), 2)
        self.assertTrue(abs(cards[-1][0] - 163) <= 2 and abs(cards[-1][1] - 401) <= 3)

    def test_no_cards(self):
        self.assertEqual(hard_list.unlocked_cards(hard_page()), [])


if __name__ == '__main__':
    unittest.main()
