"""Empty raid presets can be filled without replacing manually configured squads."""
import ast
from pathlib import Path
from types import SimpleNamespace
import re
import os
import time
import unittest
from unittest.mock import Mock

source = ast.parse((Path(__file__).resolve().parents[1] / 'module/solo_raid/challenge.py').read_text(encoding='utf-8'))

class Guard:
    def __init__(self, device, *args, **kwargs):
        self.device = device
    def check(self):
        if self.device.frames > 15:
            raise AssertionError('Formation did not terminate')

class Timer:
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        return self
    def reached(self):
        return True

namespace = dict(WaitGuard=Guard, Timer=Timer, logger=Mock(), FIGHT_HISTORY='board',
                 SAVE_TEAM='save', TEAM_NIKKE_AUTO='auto', GameStuckError=RuntimeError)
node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'fill_empty_raid_team')
exec(compile(ast.Module(body=[node], type_ignores=[]), 'formation', 'exec'), namespace)

class Runner:
    fill = namespace['fill_empty_raid_team']
    def __init__(self, state='board', empty=True, eligible=True, unavailable=False):
        self.state, self.empty, self.eligible = state, empty, eligible
        self.unavailable = unavailable
        self.actions, self.frames = [], 0
        self.device = SimpleNamespace(screenshot=self.screenshot, click_xy=self.click_xy,
                                      click=self.click, sleep=lambda _: None, frames=0)
    def screenshot(self):
        self.frames += 1
        self.device.frames = self.frames
    def appear(self, button, **kwargs):
        return button == 'board' and self.state == 'board'
    def raid_team_empty(self, **kwargs):
        return self.empty
    def raid_editor_visible(self):
        return self.state == 'editor'
    def raid_team_unavailable(self, **kwargs):
        return self.unavailable
    def active_raid_team(self, **kwargs):
        return 4
    def click_xy(self, x, y):
        if x == 310:
            self.actions.append('reset')
            self.empty, self.unavailable = True, False
            return
        self.actions.append('open' if y == 968 else 'back')
        self.state = 'editor' if y == 968 else 'board'
    def click(self, button):
        self.actions.append(button)
        if button == 'auto' and self.eligible:
            self.empty = False
    def appear_then_click(self, button, **kwargs):
        self.actions.append(button)
        if button == 'save':
            self.state = 'board'
        return True

class FormationTests(unittest.TestCase):
    def test_empty_squad_is_auto_arranged_and_saved_before_return(self):
        runner = Runner()
        self.assertTrue(runner.fill())
        self.assertEqual(runner.actions, ['open', 'auto', 'save'])
        self.assertEqual(runner.state, 'board')
        self.assertFalse(runner.empty)
    def test_existing_squad_is_not_replaced(self):
        runner = Runner(empty=False)
        self.assertFalse(runner.fill())
        self.assertEqual(runner.actions, [])
    def test_existing_editor_is_saved_without_auto_or_reset(self):
        runner = Runner(state='editor', empty=False)
        self.assertTrue(runner.fill(editor=True))
        self.assertEqual(runner.actions, ['save'])
    def test_no_eligible_characters_leaves_empty_preset_and_returns(self):
        runner = Runner(eligible=False)
        self.assertFalse(runner.fill())
        self.assertEqual(runner.actions, ['open', 'auto', 'back'])
        self.assertEqual(runner.state, 'board')
        self.assertTrue(runner.empty)
    def test_empty_editor_is_also_supported(self):
        runner = Runner(state='editor')
        self.assertTrue(runner.fill(editor=True))
        self.assertEqual(runner.actions, ['auto', 'save'])
    def test_fully_unavailable_preset_can_be_rebuilt_after_existing_squads(self):
        runner = Runner(empty=False, unavailable=True)
        self.assertTrue(runner.fill(replace_unavailable=True))
        self.assertEqual(runner.actions, ['open', 'reset', 'auto', 'save'])
    def test_preset_mismatch_prevents_resetting_another_squad(self):
        runner = Runner(empty=False, unavailable=True)
        runner.active_raid_team = lambda editor=False: 3 if editor else 4
        with self.assertRaises(RuntimeError):
            runner.fill(replace_unavailable=True)
        self.assertEqual(runner.actions, ['open'])

    def test_valid_existing_squad_is_retained_even_in_repair_mode(self):
        runner = Runner(empty=False)
        self.assertFalse(runner.fill(replace_unavailable=True))
        self.assertEqual(runner.actions, [])
    def test_unavailable_squad_requires_explicit_repair_mode(self):
        runner = Runner(empty=False, unavailable=True)
        self.assertFalse(runner.fill())
        self.assertEqual(runner.actions, [])

    def test_unconfirmed_battle_board_never_receives_input(self):
        runner = Runner(state='other')
        self.assertFalse(runner.fill())
        self.assertEqual(runner.actions, [])

class RaidResultTests(unittest.TestCase):
    def load(self, name, texts):
        ns = dict(re=re, logger=Mock(), Ocr=Mock(return_value=SimpleNamespace(ocr=Mock(side_effect=[{'text': t} for t in texts]))))
        node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), ns)
        return ns[name]
    def test_settling_text_detected_only_when_present(self):
        # 本期突击结束后中央显示「结算中」，按钮全灰；其他画面不能误判
        for text, expected in (('结算中', True), ('结 算 中', True), ('', False), ('挑战', False)):
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            self.assertEqual(self.load('raid_settling', [text])(runner), expected)

    def test_round_battle_counter_is_not_daily_counter(self):
        for text, expected in (('剩余战斗次数2/5', 2), ('0／5', 0), ('2/3', None), ('25', None), ('6/5', None)):
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            self.assertEqual(self.load('raid_battles_remaining', [text])(runner), expected)
    def test_reset_notice_requires_both_message_and_confirm_label(self):
        node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == 'handle_raid_end_notice')
        for body, label, expected in (('挑战结束。记录已被重置。', '确认', True),
                                       ('挑战结束。纪录已被重置。', '确认', True),
                                       ('挑战结束', '确认', False),
                                       ('挑战结束。记录已被重置。', '取消', False)):
            reports = []
            ocr = Mock(return_value=SimpleNamespace(ocr=Mock(side_effect=[{'text': body}, {'text': label}])))
            ns = dict(re=re, Ocr=ocr, logger=Mock(), report=lambda *a, **kw: reports.append(a))
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'notice', 'exec'), ns)
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            self.assertEqual(ns['handle_raid_end_notice'](runner), expected)
            self.assertEqual(runner.device.click_xy.call_count, int(expected))
            if expected:
                self.assertEqual(reports[-1][0], 'blocked')
                runner.device.click_xy.assert_called_once_with(360, 800)

    def test_result_exit_distinguishes_boss_defeat_from_remaining_hp(self):
        for hp in (0, 26840238):
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            fn = self.load('dismiss_raid_result', ['点击任意处进行下一步', f'HP REMAINING {hp:,}/157,017,558 TOTAL DAMAGE'])
            self.assertTrue(fn(runner))
            self.assertEqual(runner._last_raid_result_hp, hp)
            runner.device.click_xy.assert_called_once_with(360, 1130)

class RaidRecoveryTests(unittest.TestCase):
    def load(self, name, namespace):
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'module/solo_raid/solo_raid.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
        return namespace[name]
    def test_confirmed_battle_refreshes_device_watchdog_but_keeps_battle_deadline(self):
        fn = self.load('check_raid_battle_progress', dict(PAUSE='pause'))
        for active in (True, False):
            runner, guard = SimpleNamespace(appear=Mock(return_value=active), device=Mock()), Mock()
            self.assertEqual(fn(runner, guard), active)
            self.assertEqual(runner.device.stuck_record_clear.call_count, int(active))
            self.assertEqual(guard.check.call_count, int(active))
            self.assertEqual(guard.progress.call_count, int(not active))
    @unittest.skipUnless(os.name == 'nt', 'Windows device watchdog')
    def test_aged_device_watchdog_is_reset_while_real_battle_deadline_still_expires(self):
        from module.device.win.device import Device
        from module.base.timer import Timer as DeviceTimer
        from module.base.watchdog import WaitGuard
        from module.exception import GameStuckError
        device = Device.__new__(Device)
        device.image, device.detect_record = None, set()
        device.stuck_timer = DeviceTimer(360, count=60).start()
        device.stuck_timer_long = DeviceTimer(480, count=180).start()
        for timer in (device.stuck_timer, device.stuck_timer_long):
            timer._current = time.time() - 600
            timer._reach_count = 1000
        runner = SimpleNamespace(appear=Mock(return_value=True), device=device)
        guard = WaitGuard(device, 'test battle', timeout=300, warn_after=180)
        fn = self.load('check_raid_battle_progress', dict(PAUSE='pause'))
        self.assertTrue(fn(runner, guard))
        self.assertFalse(device.stuck_record_check())
        guard.started -= 301
        with self.assertRaises(GameStuckError):
            fn(runner, guard)

    def test_retry_resumes_active_battle_without_navigating_away(self):
        error = type('Unavailable', (Exception,), {})
        fn = self.load('enter_normal_raid', dict(PAUSE='pause', page_main='main', logger=Mock(), WaitGuard=lambda *a, **kw: Mock(),
                       SoloRaidTeamUnavailable=error, SoloRaidIsUnavailable=error, NoOpportunityRemain=error))
        runner = SimpleNamespace(device=Mock(), appear=Mock(return_value=True), solo_raid=Mock(),
                                 ui_ensure=Mock(), ensure_into_soloraid=Mock(),
                                 config=SimpleNamespace(SoloRaid_Challenge=False, task_delay=Mock()))
        fn(runner)
        runner.solo_raid.assert_called_once()
        runner.ui_ensure.assert_not_called()
        runner.ensure_into_soloraid.assert_not_called()

    def test_untemplated_event_page_returns_home_before_entry(self):
        # Event 任务结束停在活动主页，接续入口要用大厅按钮离开，而不是等到 45 秒超时
        error = type('Unavailable', (Exception,), {})
        fn = self.load('enter_normal_raid', dict(PAUSE='pause', page_main='main', logger=Mock(), WaitGuard=lambda *a, **kw: Mock(),
                       GOTO_MAIN='home', MAIN_CHECK='main-check', FIGHT_HISTORY='history', SOLO_RAID_CHECK='raid',
                       END_FIGHTING='end', SoloRaidTeamUnavailable=error))
        state = {'page': 'event'}
        def click_home(button, **kw):
            if button == 'home' and state['page'] == 'event':
                state['page'] = 'main-check'
                return True
            return False
        runner = SimpleNamespace(device=Mock(), appear=lambda button, **kw: button == state['page'],
                                 appear_then_click=Mock(side_effect=click_home), handle_raid_end_notice=Mock(return_value=False),
                                 raid_editor_visible=Mock(return_value=False), dismiss_raid_result=Mock(return_value=False),
                                 ui_pages=[], ui_additional=Mock(), ensure_into_soloraid=Mock(return_value='entered'))
        self.assertEqual(fn(runner), 'entered')
        runner.ui_additional.assert_not_called()


if __name__ == '__main__':
    unittest.main()
