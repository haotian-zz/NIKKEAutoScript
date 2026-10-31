"""大厅 COOP 图标只在左列找：2026-10-31 协同作战结束后，旧的细条模板全屏搜索误点了顶部资源栏 5 次。"""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def coop_clicks():
    tree = ast.parse((ROOT / 'module/coop/coop.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, 'attr', '') in ('appear', 'appear_then_click')
                and node.args and getattr(node.args[0], 'id', '') == 'COOP'):
            yield {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


class CoopIconEntryTest(unittest.TestCase):
    def test_icon_template_sits_in_lobby_left_column(self):
        tree = ast.parse((ROOT / 'module/coop/assets.py').read_text(encoding='utf-8'))
        coop = next(node.value for node in tree.body
                    if isinstance(node, ast.Assign) and node.targets[0].id == 'COOP')
        area = ast.literal_eval(next(kw.value for kw in coop.keywords if kw.arg == 'area'))['zh-CN']
        self.assertLess(area[2], 110)
        # 图标本体（logo），不是只有几像素高的细条
        self.assertGreaterEqual(area[3] - area[1], 30)

    def test_icon_search_stays_in_left_column(self):
        clicks = list(coop_clicks())
        self.assertTrue(clicks)
        for kwargs in clicks:
            self.assertNotEqual(kwargs.get('static', True), False)
            offset = kwargs.get('offset')
            self.assertIsInstance(offset, tuple)
            self.assertLessEqual(offset[0], 30)
            self.assertGreaterEqual(offset[1], 250)


if __name__ == '__main__':
    unittest.main()
