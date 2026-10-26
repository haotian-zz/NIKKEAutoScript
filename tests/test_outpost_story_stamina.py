"""前哨剧情列表里耐力用完时停下，不在「开始 → 确认」之间空转到 GameStuck。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = (Path(__file__).resolve().parents[1] / 'module/outpost/outpost_story.py').read_text()
cls = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'OutpostStory'][0]
body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ('story_view', 'close_story_list')]


class NoTimesRemaining(Exception):
    pass


class FakeTimer:
    def __init__(self, *args):
        pass

    def start(self):
        return self

    def reached(self):
        return False


namespace = {'Timer': FakeTimer, 'NoTimesRemaining': NoTimesRemaining, 'logger': SimpleNamespace(
    hr=lambda *a, **k: None, info=lambda *a, **k: None, warning=lambda *a, **k: None)}
for name in ('STORY_OPEN', 'STORY_DIALOG_BOX', 'STORY_CLOSE', 'STORY_START', 'STORY_START_CHECK',
             'STORY_START_CONFIRM', 'SKIP', 'REWARD'):
    namespace[name] = name
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'outpost_story', 'exec'), namespace)


class Runner:
    CONFIRM_WITHOUT_STORY_LIMIT = 3
    story_view = namespace['story_view']
    close_story_list = namespace['close_story_list']

    def __init__(self, frames):
        self.frames = list(frames)
        self.frame = set()
        self.clicks = []
        self.device = SimpleNamespace(screenshot=self.screenshot, click_record_clear=lambda: None,
                                      stuck_record_clear=lambda: None)

    def check_remaining(self):
        pass

    def screenshot(self):
        if not self.frames:
            raise AssertionError('loop did not stop')
        self.frame = self.frames.pop(0)

    def appear(self, button, **kwargs):
        return button in self.frame

    def appear_then_click(self, button, **kwargs):
        if button in self.frame:
            self.clicks.append(button)
            return True
        return False


CONFIRM = {'STORY_CLOSE', 'STORY_START_CHECK', 'STORY_START_CONFIRM'}
LIST = {'STORY_CLOSE', 'STORY_START'}


class OutpostStoryStaminaTest(unittest.TestCase):
    def test_confirm_that_never_enters_story_stops_and_closes_list(self):
        runner = Runner([CONFIRM, LIST] * 3 + [LIST, LIST, set()])
        with self.assertRaises(NoTimesRemaining):
            runner.story_view()
        self.assertEqual(runner.clicks.count('STORY_START_CONFIRM'), 3)
        self.assertEqual(runner.clicks[-1], 'STORY_CLOSE')

    def test_entering_story_resets_counter(self):
        frames = [CONFIRM, CONFIRM, {'SKIP'}, CONFIRM, CONFIRM, {'REWARD'}, CONFIRM, CONFIRM, CONFIRM, LIST, set()]
        runner = Runner(frames)
        with self.assertRaises(NoTimesRemaining):
            runner.story_view()
        self.assertEqual(runner.clicks.count('STORY_START_CONFIRM'), 7)


if __name__ == '__main__':
    unittest.main()
