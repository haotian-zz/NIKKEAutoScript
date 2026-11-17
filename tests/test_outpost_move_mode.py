"""前哨剧情：误入「移动建筑」模式要点取消退出；剧情条目图标的低分误匹配不再点。

实录 2026-11-16：确认框弹着时把 0.77 分的地图块当成剧情条目点了，游戏进了移动建筑模式，
脚本空等 6 分钟才 GameStuck 重启游戏。
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
STORY_SOURCE = (ROOT / 'module/outpost/outpost_story.py').read_text(encoding='utf-8')
UI_SOURCE = (ROOT / 'module/ui/ui.py').read_text(encoding='utf-8')
ASSETS_SOURCE = (ROOT / 'module/handler/assets.py').read_text(encoding='utf-8')
CANCEL_AREA = (276, 1192, 404, 1230)


def class_methods(source, class_name, names):
    cls = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    return [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]


def class_constant(source, class_name, name):
    cls = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == class_name][0]
    for node in cls.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value.value
    raise AssertionError(f'{name} not defined')


class NoTimesRemaining(Exception):
    pass


class NoStoryRemaining(Exception):
    pass


namespace = {
    'NoTimesRemaining': NoTimesRemaining,
    'NoStoryRemaining': NoStoryRemaining,
    'logger': SimpleNamespace(hr=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None),
}
for name in ('STORY_OPEN', 'STORY_DIALOG_BOX', 'STORY_CLOSE', 'STORY_START', 'STORY_START_CHECK',
             'STORY_START_CONFIRM', 'SKIP', 'REWARD', 'STORY_REMAINING_0', 'STORY_TIMES_0'):
    namespace[name] = name
body = class_methods(STORY_SOURCE, 'OutpostStory', ('story_view', 'check_remaining'))
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'outpost_story', 'exec'), namespace)


class Runner:
    """frames 里每帧是 {按钮: 分数}；appear 按传入阈值（默认 0.74）判断，和 Button.match 一样。"""

    CONFIRM_WITHOUT_STORY_LIMIT = 3
    DIALOG_BOX_THRESHOLD = class_constant(STORY_SOURCE, 'OutpostStory', 'DIALOG_BOX_THRESHOLD')
    story_view = namespace['story_view']
    check_remaining = namespace['check_remaining']

    def __init__(self, frames):
        self.frames = list(frames)
        self.frame = {}
        self.clicks = []
        self.device = SimpleNamespace(screenshot=self.screenshot, click_record_clear=lambda: None,
                                      stuck_record_clear=lambda: None)

    def screenshot(self):
        if not self.frames:
            raise StopIteration
        self.frame = self.frames.pop(0)

    def appear(self, button, threshold=0.74, **kwargs):
        return self.frame.get(button, 0.) > threshold

    def appear_then_click(self, button, threshold=0.74, **kwargs):
        if self.appear(button, threshold=threshold):
            self.clicks.append(button)
            return True
        return False

    def handle_outpost_move_mode(self):
        # 与 InfoHandler.handle_outpost_move_mode 相同：看到底部「取消」就点
        return self.appear_then_click('OUTPOST_MOVE_CANCEL', threshold=0.85)

    def close_story_list(self):
        pass


def run_story(frames):
    runner = Runner(frames)
    runner.check_remaining = lambda: None
    try:
        runner.story_view()
    except StopIteration:
        pass
    return runner.clicks


MOVE_MODE = {'OUTPOST_MOVE_CANCEL': 0.99, 'STORY_DIALOG_BOX': 0.82}


class MoveModeRecoveryTest(unittest.TestCase):
    def test_story_loop_cancels_move_mode_before_anything_else(self):
        clicks = run_story([MOVE_MODE, {'STORY_OPEN': 0.99}])
        self.assertEqual(clicks, ['OUTPOST_MOVE_CANCEL', 'STORY_OPEN'])

    def test_check_remaining_cancels_move_mode(self):
        runner = Runner([MOVE_MODE, {'STORY_OPEN': 0.99}])
        runner.check_remaining()
        self.assertEqual(runner.clicks, ['OUTPOST_MOVE_CANCEL'])

    def test_ui_additional_handles_move_mode(self):
        method = class_methods(UI_SOURCE, 'UI', ('ui_additional',))[0]
        calls = {n.func.attr for n in ast.walk(method) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertIn('handle_outpost_move_mode', calls)


class DialogBoxThresholdTest(unittest.TestCase):
    def test_low_score_map_patch_is_not_clicked(self):
        # 实录分数：误点 0.740～0.817，真条目 0.934 以上
        for score in (0.74, 0.767, 0.793, 0.817):
            self.assertNotIn('STORY_DIALOG_BOX', run_story([{'STORY_DIALOG_BOX': score + 0.0001}] * 3), score)

    def test_real_story_entry_is_clicked(self):
        for score in (0.934, 0.952, 0.994):
            self.assertIn('STORY_DIALOG_BOX', run_story([{'STORY_DIALOG_BOX': score}]), score)

    def test_no_dialog_click_while_confirm_popup_is_up(self):
        # 23:47:38 那一帧：确认框还在动画，STORY_START_CHECK 没出来，条目图标 0.95 也不能点
        popup = {'STORY_START_CONFIRM': 0.99, 'STORY_CLOSE': 0.99, 'STORY_DIALOG_BOX': 0.95}
        self.assertNotIn('STORY_DIALOG_BOX', run_story([popup] * 3))

    def test_low_score_does_not_block_opening_list(self):
        clicks = run_story([{'STORY_DIALOG_BOX': 0.8, 'STORY_OPEN': 0.99}])
        self.assertEqual(clicks, ['STORY_OPEN'])


class CancelAssetTest(unittest.TestCase):
    def test_asset_entry_matches_image(self):
        line = [l for l in ASSETS_SOURCE.splitlines() if l.startswith('OUTPOST_MOVE_CANCEL = ')]
        self.assertEqual(len(line), 1)
        self.assertIn(f"'zh-CN': {CANCEL_AREA}", line[0])
        self.assertTrue((ROOT / 'assets/zh-CN/handler/OUTPOST_MOVE_CANCEL.png').exists())

    def test_template_matches_real_frame_and_rejects_map(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('cv2 not installed')
        asset = cv2.imread(str(ROOT / 'assets/zh-CN/handler/OUTPOST_MOVE_CANCEL.png'))
        x1, y1, x2, y2 = CANCEL_AREA
        template = asset[y1:y2, x1:x2]

        def similarity(patch, left, top):
            canvas = np.zeros((1280, 720, 3), np.uint8)
            canvas[top:top + patch.shape[0], left:left + patch.shape[1]] = patch
            # Button.match(offset=(20, 20))：只在按钮位置上下左右 20 像素内找
            search = canvas[y1 - 20:y2 + 20, x1 - 20:x2 + 20]
            return cv2.minMaxLoc(cv2.matchTemplate(template, search, cv2.TM_CCOEFF_NORMED))[1]

        fixtures = Path(__file__).parent / 'fixtures'
        # 正样本取自卡住的第一帧（模板取自 6 分钟后的另一帧）
        stuck = cv2.imread(str(fixtures / 'outpost_move_mode_bottom.png'))
        self.assertGreater(similarity(stuck, 216, 1132), 0.95)
        # 负样本：同一位置换成基地地图
        world = cv2.imread(str(fixtures / 'outpost_map_patch.png'))
        self.assertLess(similarity(world, 216, 1132), 0.5)


if __name__ == '__main__':
    unittest.main()
