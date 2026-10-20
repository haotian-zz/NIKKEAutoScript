"""Regression for returning from a dispatch board through an unknown building panel."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


source = ast.parse((Path(__file__).resolve().parents[1] / 'module/ui/ui.py').read_text())
method = next(node for node in ast.walk(source)
              if isinstance(node, ast.FunctionDef) and node.name == 'ui_additional')


class NavigationPopupTests(unittest.TestCase):
    def test_only_paired_facility_controls_close_before_rewards(self):
        for back, enter in ((True, True), (True, False), (False, True), (False, False)):
            with self.subTest(back=back, enter=enter):
                ns = dict(FACILITY_BACK='back', FACILITY_ENTER='enter')
                exec(compile(ast.Module(body=[method], type_ignores=[]), 'ui_additional', 'exec'), ns)
                handler = SimpleNamespace(appear=Mock(side_effect=lambda button, **kw: back if button == 'back' else enter),
                                          appear_then_click=Mock(return_value=True),
                                          handle_level_up=Mock(return_value=False),
                                          handle_reward=Mock(return_value=True))
                self.assertTrue(ns['ui_additional'](handler))
                if back and enter:
                    handler.appear_then_click.assert_called_once_with('back', offset=10, threshold=0.9, interval=1)
                    handler.handle_reward.assert_not_called()
                else:
                    handler.appear_then_click.assert_not_called()
                    handler.handle_reward.assert_called_once()


    def test_known_dispatch_or_outpost_page_does_not_click_gray_login_claim(self):
        for current in ('commission', 'outpost', 'login-reward'):
            with self.subTest(current=current):
                ns = dict(FACILITY_BACK='back', FACILITY_ENTER='enter', LOGIN_PAGE_CHECK='login',
                          LOGO_PAGE_CHECK='logo', CONFIRM_A='a', CONFIRM_B='b', CONFIRM_C='c',
                          page_commission=SimpleNamespace(check_button='commission'),
                          page_outpost=SimpleNamespace(check_button='outpost'))
                exec(compile(ast.Module(body=[method], type_ignores=[]), 'ui_additional', 'exec'), ns)
                handler = Mock()
                for name in ('handle_level_up', 'handle_reward', 'handle_paid_gift', 'handle_announcement',
                             'handle_server', 'handle_download', 'handle_system_error',
                             'handle_system_maintenance', 'handle_login', 'appear_then_click'):
                    getattr(handler, name).return_value = False
                handler.handle_login_reward.return_value = True
                handler.appear.side_effect = lambda button, **kw: button == current
                ns['ui_additional'](handler)
                if current == 'login-reward':
                    handler.handle_login_reward.assert_called_once()
                else:
                    handler.handle_login_reward.assert_not_called()


if __name__ == '__main__':
    unittest.main()
