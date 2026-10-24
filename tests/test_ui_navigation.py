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
        """实录 2026-10-24：防御前哨奖励页的「获得奖励」也不是登录奖励弹窗。"""
        for current in ('commission', 'outpost', 'reward', 'login-reward'):
            with self.subTest(current=current):
                ns = dict(FACILITY_BACK='back', FACILITY_ENTER='enter', LOGIN_PAGE_CHECK='login',
                          LOGO_PAGE_CHECK='logo', CONFIRM_A='a', CONFIRM_B='b', CONFIRM_C='c',
                          page_commission=SimpleNamespace(check_button='commission'),
                          page_outpost=SimpleNamespace(check_button='outpost'),
                          page_reward=SimpleNamespace(check_button='reward'))
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


class NavigationHomeFallbackTests(unittest.TestCase):
    def navigate(self, known_cooldown=False, loading=False, home_reached=True):
        class Page:
            def __init__(self, check):
                self.check_button, self.links, self.parent = check, {}, None
        target, main = Page('target'), Page('main')
        main.links[target] = 'go-target'
        state = {'page': 'main' if known_cooldown else 'unknown', 'loops': 0, 'checks': 0}
        if loading:
            state['page'] = 'loading'
        def screenshot():
            if state['page'] == 'loading':
                state['page'] = 'target'
        clicks = []
        def check_guard():
            state['loops'] += 1
            if state['loops'] > 10:
                raise AssertionError('Navigation did not finish')
        def appear(button, **kwargs):
            if button == 'home':
                return state['page'] != 'target'
            if kwargs.get('interval') and known_cooldown:
                state['checks'] += 1
                if state['checks'] < 3:
                    return False
            return button == state['page']
        def click(button):
            clicks.append(button)
            state['page'] = 'target' if button == 'go-target' else 'main'
        def click_home(button, **kwargs):
            if appear(button):
                click(button)
                return True
            return False
        timer = SimpleNamespace(start=lambda: timer, reset=lambda: timer, reached=lambda: True)
        home_timer = SimpleNamespace(start=lambda: home_timer, reset=lambda: home_timer, reached=lambda: home_reached)
        method = next(node for node in ast.walk(source)
                      if isinstance(node, ast.FunctionDef) and node.name == 'ui_goto')
        namespace = dict(logger=Mock(), Timer=lambda *a, **kw: home_timer if a[0] == 'home-wait' else timer,
                         GOTO_MAIN='home', UNKNOWN_HOME_SECONDS='home-wait',
                         WaitGuard=lambda *a, **kw: SimpleNamespace(check=check_guard))
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'ui_goto', 'exec'), namespace)
        ui = SimpleNamespace(ui_pages=[main, target], device=SimpleNamespace(screenshot=screenshot, click=click),
                             appear=appear, appear_then_click=click_home, ui_additional=lambda: False)
        namespace['ui_goto'](ui, target)
        return clicks

    def test_unknown_selection_page_can_return_home_then_continue(self):
        self.assertEqual(self.navigate(), ['home', 'go-target'])

    def test_known_page_cooldown_does_not_trigger_home_fallback(self):
        self.assertEqual(self.navigate(known_cooldown=True), ['go-target'])

    def test_loading_frame_before_page_check_does_not_click_home(self):
        # 方舟切到竞技场时，底栏大厅按钮比竞技场标识早一帧出现；此时点大厅会把刚到的页面带回大厅
        self.assertEqual(self.navigate(loading=True, home_reached=False), [])


if __name__ == '__main__':
    unittest.main()
