"""活动挑战打输时退出战败页，不再中断整个日常。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = (Path(__file__).resolve().parents[1] / 'module/event/challenge.py').read_text()
methods = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)][0].body
body = [node for node in methods if isinstance(node, ast.FunctionDef) and node.name == 'leave_failed_battle']


class Takeover(Exception):
    pass


class FakeTimer:
    def __init__(self, *args):
        self.ticks = 0

    def start(self):
        return self

    def reached(self):
        self.ticks += 1
        return self.ticks > 5


namespace = {'Timer': FakeTimer, 'OPERATION_FAILED': 'failed', 'BACK': 'back', 'RequestHumanTakeover': Takeover}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'challenge', 'exec'), namespace)


class Runner:
    leave_failed_battle = namespace['leave_failed_battle']

    def __init__(self, failed_frames):
        self.failed_frames = failed_frames
        self.clicks = []
        self.device = SimpleNamespace(screenshot=self.screenshot)

    def screenshot(self):
        self.failed_frames -= 1

    def appear(self, button, offset=0):
        return self.failed_frames >= 0

    def appear_then_click(self, button, offset=0, interval=0):
        self.clicks.append(button)
        return True


class LeaveFailedBattleTest(unittest.TestCase):
    def test_clicks_back_until_failed_page_disappears(self):
        runner = Runner(failed_frames=2)
        runner.leave_failed_battle()
        self.assertEqual(runner.clicks, ['back', 'back'])

    def test_failed_page_that_never_leaves_requests_takeover(self):
        runner = Runner(failed_frames=100)
        with self.assertRaises(Takeover):
            runner.leave_failed_battle()

    def test_failure_returns_to_event_instead_of_stopping_daily(self):
        branch = source[source.index('if self.appear(OPERATION_FAILED'):]
        branch = branch[:branch.index('self.back_to_event()\n        logger.info')]
        self.assertIn('self.leave_failed_battle()', branch)
        self.assertIn('return False', branch)
        self.assertNotIn('raise RequestHumanTakeover', branch)


if __name__ == '__main__':
    unittest.main()
