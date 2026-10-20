"""Startup is shared by all task entry points without changing the current game page."""
import ast
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]

def method(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


class StartupTests(unittest.TestCase):
    def test_closed_game_starts_but_open_game_keeps_its_page(self):
        assets = SimpleNamespace(LOGIN_PAGE_CHECK='title', LOGO_PAGE_CHECK='logo', LOGIN_CHECK='touch', LOGIN_CHECK_B='touch-b')
        for running, pending, title, expected in ((False, False, False, 'start'), (True, False, False, 'continue'),
                                                  (True, True, False, 'login'), (True, False, True, 'login')):
            with self.subTest(running=running, pending=pending, title=title):
                login = Mock();login.appear.return_value = title
                script = SimpleNamespace(device=Mock(login_required=pending), config=Mock(), start=Mock())
                script.device.app_is_running.return_value = running
                fn = method('main.py', 'ensure_game_ready', dict(logger=Mock()))
                with patch.dict(sys.modules, {'module.handler': SimpleNamespace(assets=assets),
                                               'module.handler.login': SimpleNamespace(LoginHandler=Mock(return_value=login))}):
                    fn(script)
                self.assertEqual(script.start.call_count, int(expected == 'start'))
                self.assertEqual(login.handle_app_login.call_count, int(expected == 'login'))
                login.app_restart.assert_not_called()
                login.ui_goto_main.assert_not_called()

    def test_independent_tasks_and_explicit_restart_skip_startup_check(self):
        import module.exception as exceptions
        for command, independent, expected in (('notify', True, 0), ('restart', False, 0),
                                                 ('event', False, 1), ('mailbox', False, 1)):
            fn = method('main.py', 'run', dict(vars(exceptions), logger=Mock(), TaskEnd=type('TaskEnd', (Exception,), {})))
            script = SimpleNamespace(is_independent_task=Mock(return_value=independent), ensure_game_ready=Mock(),
                                     device=Mock(), **{command: Mock()})
            with self.subTest(command=command):
                self.assertTrue(fn(script, command, skip_first_screenshot=True))
                self.assertEqual(script.ensure_game_ready.call_count, expected)

    def test_scheduler_checks_startup_even_before_future_tasks_and_keeps_it_hidden(self):
        fn = method('main.py', 'prepare_startup', dict(logger=Mock(), datetime=datetime, del_cached_property=Mock()))
        for command, independent, expected in (('Reward', False, 1), ('Notify', True, 0)):
            script = SimpleNamespace(config=SimpleNamespace(data={command: {'Scheduler': {'Enable': True}}}),
                                     is_independent_task=Mock(return_value=independent), serial_report_due=Mock(),
                                     serial_wait_turn=Mock(return_value=True), driver_wait_turn=Mock(return_value=True),
                                     run=Mock(), execution=Mock())
            fn(script)
            self.assertEqual(script.run.call_count, expected)
            script.execution.begin.assert_not_called()
            if expected:
                script.run.assert_called_once_with('startup', skip_first_screenshot=True)

    def test_failed_hidden_startup_keeps_scheduled_restart_recovery_enabled(self):
        fn = method('main.py', 'prepare_startup', dict(logger=Mock(), datetime=datetime, del_cached_property=Mock()))
        script = SimpleNamespace(config=SimpleNamespace(data={'Reward': {'Scheduler': {'Enable': True}}}),
                                 is_independent_task=Mock(return_value=False), serial_report_due=Mock(),
                                 serial_wait_turn=Mock(return_value=True), driver_wait_turn=Mock(return_value=True),
                                 run=Mock(return_value=False), is_first_task=True)
        fn(script)
        self.assertFalse(script.is_first_task)

    def test_login_flag_is_cleared_only_after_confirmed_lobby(self):
        fn = method('module/handler/login.py', 'handle_app_login', dict(logger=Mock(),
                    GameStuckError=type('Stuck', (Exception,), {}),
                    GameTooManyClickError=type('Clicks', (Exception,), {}),
                    RequestHumanTakeover=type('Takeover', (Exception,), {})))
        script = SimpleNamespace(device=Mock(login_required=True), _handle_app_login=Mock())
        self.assertTrue(fn(script))
        self.assertIs(script.device.login_required, False)
        script.device.app_stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
