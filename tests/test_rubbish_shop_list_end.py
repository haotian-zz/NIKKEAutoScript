"""废铁商店：底部颜色认不出时，滚不动也要当成列表到底，接着处理后面的商品。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'module/shop/rubbish.py').read_text(encoding='utf-8')
cls = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'RubbishShop'][0]
body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'swipe_and_purchase']


class Stuck(Exception):
    pass


class FakeTimer:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def reached(self):
        return True


namespace = {'Timer': FakeTimer, 'GameStuckError': Stuck, 'SelectedGrids': object, 'cv2': cv2,
             'BUY_ALL': 'buy_all', 'END_LIST_CHECK': 'end_list', 'PURCHASE_CHECK': 'purchase',
             'logger': SimpleNamespace(attr=lambda *a: None, info=lambda *a: None, warning=lambda *a: None)}
exec(compile(ast.Module(body=body, type_ignores=[]), 'rubbish', 'exec'), namespace)


class FakeShop:
    swipe_and_purchase = namespace['swipe_and_purchase']

    def __init__(self, positions, bottom, end_list_visible=False):
        # positions: 商品名 → 第几屏能看到；bottom: 最多能滚到第几屏
        self.positions = positions
        self.bottom = bottom
        self.end_list_visible = end_list_visible
        self.pos = 0
        self.popup = None
        self.bought = []
        self.scrolls = 0
        self.device = SimpleNamespace(screenshot=self.screenshot, image=None)

    def screenshot(self):
        self.device.image = np.full((1280, 720, 3), 20 * self.pos, dtype=np.uint8)

    def appear(self, button, **kwargs):
        if button == 'purchase':
            return self.popup is not None
        if button == 'end_list':
            return self.end_list_visible and self.pos == self.bottom
        return button == 'buy_all'

    def appear_then_click(self, button, **kwargs):
        if self.positions.get(button) == self.pos and button not in self.bought:
            self.popup = button
            return True
        return False

    def handle_purchase(self, button=None, skip_first_screenshot=True):
        self.bought.append(self.popup)
        self.popup = None

    def ensure_sroll_to_top(self, *args, **kwargs):
        self.pos = 0

    def ensure_sroll(self, *args, **kwargs):
        self.scrolls += 1
        self.pos = min(self.pos + 1, self.bottom)


def products(*names):
    return [SimpleNamespace(name=n, button=n) for n in names]


class RubbishShopListEndTest(unittest.TestCase):
    def test_list_end_by_no_movement_when_color_check_misses(self):
        shop = FakeShop({'GEM': 0, 'CORE_DUST_CASE': 3, 'CREDIT': 4}, bottom=4)
        shop.swipe_and_purchase(products('GEM', 'CORE_DUST_CASE', 'CREDIT_CASE', 'CREDIT'))
        self.assertEqual(shop.bought, ['GEM', 'CORE_DUST_CASE', 'CREDIT'])
        # 每个商品最多滚到底再多一下，远少于卡死上限
        self.assertLessEqual(shop.scrolls, 4 * (4 + 1))

    def test_end_list_color_still_ends_scan(self):
        shop = FakeShop({}, bottom=3, end_list_visible=True)
        shop.swipe_and_purchase(products('GEM'))
        self.assertEqual(shop.scrolls, 3)


if __name__ == '__main__':
    unittest.main()
