"""竖屏章节列表困难页：列表锚点、困难页签与已解锁卡片识别。"""

import unittest
from pathlib import Path

import cv2
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

    def test_list_visible_with_story_badge(self):
        """实录 2026-10-24：有新剧情时「剧情」页签右上角叠红色 N 角标，列表仍要认出来。"""
        crop = cv2.imread(str(Path(__file__).resolve().parent / 'fixtures/chapter_list_story_badge.png'))
        image = np.zeros((1280, 720, 3), np.uint8)
        x1, y1, x2, y2 = hard_list.LIST_TAB_REGION
        image[y1:y2, x1:x2] = crop
        self.assertTrue(hard_list.list_visible(image))
        self.assertFalse(hard_list.hard_tab_active(image))

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


def counter_page():
    """实录 2026-10-25 困难页（第 7～9 章计数）放回 720×1280 原位；返回画面与三张卡片的标题框中心。"""
    crop = cv2.imread(str(Path(__file__).resolve().parent / 'fixtures/hard_list_counters.png'))
    image = np.zeros((1280, 720, 3), np.uint8)
    image[180:180 + crop.shape[0], 40:40 + crop.shape[1]] = crop
    return image, {7: (557, 196), 8: (163, 315), 9: (557, 435)}


class CounterTests(unittest.TestCase):
    """卡片进度：空心描边计数靠斜杠两侧字形比较判断是否满，OCR 只报剩余数。"""

    def test_full_counters_match_and_partial_does_not(self):
        image, cards = counter_page()
        for chapter in (7, 8):
            progress = hard_list.card_progress(image, cards[chapter])
            self.assertTrue(progress['complete'], (chapter, progress))
        progress = hard_list.card_progress(image, cards[9])
        self.assertFalse(progress['complete'], progress)
        self.assertLess(progress['stage']['same'], .8)

    def test_ocr_reads_only_veto_full_and_complete_stamp_wins(self):
        """实录：OCR 把 27/27 读成 27127（不否决），读到明确的 n<N 才否决；卡片写 COMPLETE 直接算满。"""
        image, cards = counter_page()
        texts = iter(['', '27127', '2/2'])
        progress = hard_list.card_progress(image, cards[8], ocr=lambda area: next(texts))
        self.assertTrue(progress['complete'])
        self.assertIsNone(progress['stage']['read'])
        texts = iter(['', '26/27', '2/2'])
        self.assertFalse(hard_list.card_progress(image, cards[8], ocr=lambda area: next(texts))['complete'])
        self.assertTrue(hard_list.card_progress(np.zeros_like(image), cards[9],
                                                ocr=lambda area: 'COMPLETE')['complete'])

    def test_missing_counter_is_unknown(self):
        image, cards = counter_page()
        self.assertIsNone(hard_list.card_progress(np.zeros_like(image), cards[9])['complete'])
        self.assertEqual(hard_list.parse_counter('·12/27'), (12, 27))
        self.assertIsNone(hard_list.parse_counter('2127'))


if __name__ == '__main__':
    unittest.main()
