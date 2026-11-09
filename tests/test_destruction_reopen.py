"""免费歼灭：歼灭弹窗被误关、退回前哨基地页时要重新打开，不能一直干等到卡死。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'module/destruction/destruction.py').read_text(encoding='utf-8')
cls = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'Destruction'][0]
body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'destroy_free']


class Stuck(Exception):
    pass


class FakeTimer:
    def __init__(self, *args, **kwargs):
        self._started = False

    def started(self):
        return self._started

    def start(self):
        self._started = True

    def reached(self):
        return True

    def clear(self):
        self._started = False


namespace = {'Timer': FakeTimer, 'DESTROY': 'destroy', 'DESTROY_CHECK': 'destroy_check',
             'DESTROY_GEM_CHECK': 'gem_check', 'REWARD_CHECK': 'reward_check',
             'REWARD_GOTO_DESTROY': 'goto_destroy',
             'logger': SimpleNamespace(hr=lambda *a: None, info=lambda *a: None)}
exec(compile(ast.Module(body=body, type_ignores=[]), 'destruction', 'exec'), namespace)


class FakeDestruction:
    destroy_free = namespace['destroy_free']

    def __init__(self, state):
        # outpost: 前哨基地页（歼灭弹窗已关）；destroy: 歼灭弹窗；result: 领奖弹窗；done: 免费次数用完的歼灭弹窗
        self.state = state
        self.reopen = 0
        self.destroyed = 0
        self.frames = 0
        self.device = SimpleNamespace(screenshot=self.screenshot)

    def screenshot(self):
        self.frames += 1
        if self.frames > 50:
            raise Stuck(self.state)

    def appear(self, button, **kwargs):
        return {
            'destroy_check': self.state in ('destroy', 'done'),
            'gem_check': self.state == 'done',
            'reward_check': self.state == 'outpost',
        }.get(button, False)

    def appear_then_click(self, button, **kwargs):
        if button == 'destroy' and self.state == 'destroy':
            self.destroyed += 1
            self.state = 'result'
            return True
        if button == 'goto_destroy' and self.state == 'outpost':
            self.reopen += 1
            self.state = 'destroy'
            return True
        return False

    def handle_level_up(self):
        return False

    def handle_reward(self, interval=1):
        if self.state == 'result':
            self.state = 'done'
            return True
        return False

    def handle_paid_gift(self):
        return False


class DestructionReopenTest(unittest.TestCase):
    def test_reopen_when_popup_closed(self):
        d = FakeDestruction('outpost')
        d.destroy_free()
        self.assertEqual((d.reopen, d.destroyed, d.state), (1, 1, 'done'))

    def test_normal_flow_does_not_reopen(self):
        d = FakeDestruction('destroy')
        d.destroy_free()
        self.assertEqual((d.reopen, d.destroyed), (0, 1))


if __name__ == '__main__':
    unittest.main()
