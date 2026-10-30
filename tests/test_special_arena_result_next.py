"""特殊竞技场新版结算页（ARENA RESULT，底部「点击进行下一步 ESC」）要点掉继续。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = (Path(__file__).resolve().parents[1] / 'module/special_arena/special_arena.py').read_text()
cls = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'SpecialArena'][0]
body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'start_competition']


class FakeTimer:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return self

    def reset(self):
        pass

    def reached(self):
        return True


namespace = {'Timer': FakeTimer, 'point2str': lambda x, y: f'({x}, {y})',
             'logger': SimpleNamespace(hr=lambda *a: None, info=lambda *a: None, warning=lambda *a: None)}
for name in ('UPDATE_POP_CHECK', 'SKIP', 'INTO_COMPETITION', 'END_COMPETITION', 'RESULT_NEXT', 'SPECIAL_ARENA_CHECK'):
    namespace[name] = name
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'special_arena', 'exec'), namespace)


class Runner:
    start_competition = namespace['start_competition']
    free_opportunity_remain = False

    def __init__(self, frames):
        self.frames = list(frames)
        self.frame = set()
        self.clicks = []
        self.battle_times = 1
        self.config = SimpleNamespace(SpecialArena_Times=1, OpponentSelection_Enable=False)
        self.device = SimpleNamespace(
            screenshot=self.screenshot, click_xy=lambda x, y: self.clicks.append((x, y)),
            handle_control_check=lambda b: None, click_record_clear=lambda: None, stuck_record_clear=lambda: None)

    def screenshot(self):
        if not self.frames:
            raise AssertionError('result page was not left')
        self.frame = self.frames.pop(0)

    def appear(self, button, **kwargs):
        return button in self.frame

    def appear_then_click(self, button, **kwargs):
        return False


class SpecialArenaResultNextTest(unittest.TestCase):
    def test_new_result_page_is_tapped_through(self):
        runner = Runner([{'RESULT_NEXT'}, {'SPECIAL_ARENA_CHECK'}])
        runner.start_competition(skip_first_screenshot=False)
        self.assertEqual(runner.clicks, [(100, 100)])
        self.assertEqual(runner.battle_times, 0)


if __name__ == '__main__':
    unittest.main()
