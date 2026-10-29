"""登录时游戏在下载资源算有进展；大厅颜色判定阈值一致。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'module/handler/login.py').read_text()
cls = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'LoginHandler'][0]
body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_handle_app_login']


class Stuck(Exception):
    pass


class FakeGuard:
    instances = []

    def __init__(self, device, label, timeout=90, warn_after=30):
        self.label = label
        self.limit = timeout // 100  # 登录 300 秒 → 3 次，下载 3600 秒 → 36 次
        self.checks = 0
        FakeGuard.instances.append(self)

    def progress(self):
        self.checks = 0

    def check(self):
        self.checks += 1
        if self.checks > self.limit:
            raise Stuck(self.label)


class FakeTimer:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        return self

    def reset(self):
        pass

    def reached(self):
        return True


namespace = {'WaitGuard': FakeGuard, 'Timer': FakeTimer, 'MAIN_CHECK': 'main', 'LOGIN_CHECK': 'login',
             'LOGIN_CHECK_B': 'login_b', 'logger': SimpleNamespace(hr=lambda *a: None, info=lambda *a: None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'login', 'exec'), namespace)


class Runner:
    _handle_app_login = namespace['_handle_app_login']

    def __init__(self, frames):
        self.frames = list(frames)
        self.frame = None
        self.device = SimpleNamespace(screenshot=self.screenshot, get_orientation=lambda: None, click=lambda b: None)

    def screenshot(self):
        self.frame = self.frames.pop(0) if self.frames else self.frame

    def appear(self, button, **kwargs):
        return button == 'main' and self.frame == 'main'

    def handle_downloading(self):
        return self.frame == 'downloading'

    def __getattr__(self, name):
        if name.startswith('handle_'):
            return lambda *a, **k: False
        raise AttributeError(name)


class LoginDownloadProgressTest(unittest.TestCase):
    def setUp(self):
        FakeGuard.instances = []

    def test_long_download_is_progress(self):
        Runner(['downloading'] * 20 + ['main'])._handle_app_login()

    def test_download_wait_is_still_bounded(self):
        with self.assertRaises(Stuck) as ctx:
            Runner(['downloading'] * 100)._handle_app_login()
        self.assertEqual(str(ctx.exception), '游戏资源下载')

    def test_no_progress_without_download_still_times_out(self):
        with self.assertRaises(Stuck) as ctx:
            Runner(['black'] * 10)._handle_app_login()
        self.assertEqual(str(ctx.exception), '游戏启动登录')


class MainCheckColorThresholdTest(unittest.TestCase):
    def test_main_check_color_threshold_matches_login(self):
        for path in (ROOT / 'module').rglob('*.py'):
            for value in re.findall(r'appear\(MAIN_CHECK, threshold=(\d+)\)', path.read_text(encoding='utf-8')):
                self.assertGreaterEqual(int(value), 20, path)


if __name__ == '__main__':
    unittest.main()
