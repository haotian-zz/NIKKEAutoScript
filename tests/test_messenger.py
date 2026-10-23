"""即时通讯任务对话：列表/对话页区分、回复选项、未读行与大厅图标识别。"""

import unittest

import numpy as np

from module.messenger import messenger

ORANGE = (10, 165, 254)
GRAY = (242, 242, 242)
WHITE = (255, 255, 255)
RED = (40, 40, 230)


def blank():
    image = np.zeros((1280, 720, 3), np.uint8)
    image[:] = (60, 60, 60)
    image[105:1170, 40:680] = GRAY
    return image


def list_view(badges=(405,)):
    """任务列表：橙色顶栏延伸到页签行，右侧红色 N 标记未读。"""
    image = blank()
    image[105:305, 40:680] = ORANGE
    for y in badges:
        image[y - 10:y + 10, 625:648] = RED
    return image


def chat_view():
    """对话页：橙色顶栏到 260，左侧白色返回箭头。"""
    image = blank()
    image[105:260, 40:680] = ORANGE
    image[192:218, 74:86] = WHITE
    return image


def add_option(image, y):
    """回复选项：橙色外框 + 白色内框。"""
    image[y:y + 90, 277:650] = ORANGE
    image[y + 15:y + 75, 300:620] = WHITE
    return image


def add_sent(image, y):
    """已发送的气泡：橙底白字，靠右。"""
    image[y:y + 64, 300:650] = ORANGE
    image[y + 25:y + 40, 330:420] = WHITE
    return image


class ViewTests(unittest.TestCase):
    def test_list_chat_other(self):
        self.assertEqual(messenger.view(list_view()), 'list')
        self.assertEqual(messenger.view(chat_view()), 'chat')
        self.assertEqual(messenger.view(blank()), 'other')

    def test_sent_bubble_under_header_keeps_chat(self):
        """实录：已发送气泡滚到顶栏下方时，中间横条变成部分橙色，仍应判为对话页。"""
        image = add_sent(chat_view(), 270)
        self.assertEqual(messenger.view(image), 'chat')


class OptionTests(unittest.TestCase):
    def test_option_detected_sent_bubble_ignored(self):
        image = add_sent(add_option(chat_view(), 1045), 600)
        options = messenger.reply_options(image)
        self.assertEqual(len(options), 1)
        x, y = options[0]
        self.assertTrue(300 < x < 620 and 1060 < y < 1120)

    def test_option_high_in_short_chat(self):
        """实录：对话很短时选项出现在 y≈590，必须照样识别（只点底部会卡住）。"""
        options = messenger.reply_options(add_option(chat_view(), 545))
        self.assertEqual(len(options), 1)
        self.assertTrue(560 < options[0][1] < 620)

    def test_no_option(self):
        self.assertEqual(messenger.reply_options(add_sent(chat_view(), 600)), [])


class UnreadTests(unittest.TestCase):
    def test_rows_from_badges(self):
        ys = messenger.unread_rows(list_view(badges=(405, 512, 619)))
        self.assertEqual(len(ys), 3)
        self.assertTrue(all(b < a for a, b in zip(ys[1:], ys)))

    def test_no_badge(self):
        self.assertEqual(messenger.unread_rows(list_view(badges=())), [])

    def test_row_below_panel_not_clicked(self):
        """面板最底下露出一半的行不点，避免点到大厅底栏。"""
        self.assertEqual(messenger.unread_rows(list_view(badges=(1150,))), [])


class TypingTests(unittest.TestCase):
    """读完的判断：既没有回复选项也没有「… ▼」气泡。"""

    def test_typing_bubble_found(self):
        image = chat_view()
        h, w = messenger.TYPING.shape[:2]
        image[1098:1098 + h, 180:180 + w] = messenger.TYPING
        self.assertTrue(messenger.typing(image))

    def test_typing_bubble_near_top_found(self):
        """对话刚打开时气泡在上方 y≈415。"""
        image = chat_view()
        h, w = messenger.TYPING.shape[:2]
        image[400:400 + h, 180:180 + w] = messenger.TYPING
        self.assertTrue(messenger.typing(image))

    def test_finished_chat_has_no_typing(self):
        self.assertFalse(messenger.typing(add_sent(chat_view(), 900)))


class IconTests(unittest.TestCase):
    def test_icon_found_anywhere_in_left_column(self):
        image = np.random.RandomState(0).randint(0, 80, (1280, 720, 3)).astype(np.uint8)
        h, w = messenger.ICON.shape[:2]
        image[400:400 + h, 24:24 + w] = messenger.ICON
        x, y = messenger.find_icon(image)
        self.assertTrue(abs(x - (24 + w // 2)) <= 1 and abs(y - (400 + h // 2)) <= 1)

    def test_icon_absent(self):
        image = np.random.RandomState(1).randint(0, 80, (1280, 720, 3)).astype(np.uint8)
        self.assertIsNone(messenger.find_icon(image))


if __name__ == '__main__':
    unittest.main()
