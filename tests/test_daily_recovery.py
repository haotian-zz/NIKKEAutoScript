"""Guard free daily attempts and recognize the current action labels without sending input."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch


def methods(path, names, namespace):
    tree = ast.parse((Path(__file__).resolve().parents[1] / path).read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace


conversation = methods('module/conversation/conversation.py', {'detail_opportunities', 'get_next_target'}, {'re': re})
interception = methods('module/interception/interception.py', {'special_button_enabled', 'run_special'}, {'re': re})


class DailyCounterTests(unittest.TestCase):
    def test_conversation_reads_explicit_fraction_and_keeps_unknown_separate(self):
        script = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
        for text, expected in [('咨询次数0/9', 0), ('剩余 3／12', 3), ('09', None), ('10/9', None), ('', None)]:
            with self.subTest(text=text), patch.dict(conversation, Ocr=Mock(return_value=SimpleNamespace(
                    ocr=lambda _: {'text': text}))):
                self.assertEqual(conversation['detail_opportunities'](script), expected)

    def test_zero_counter_stops_before_cycling_through_more_characters(self):
        no_attempts = type('NoOpportunitiesRemain', (Exception,), {})
        script = SimpleNamespace(device=Mock(), detail_opportunities=lambda: 0)
        with patch.dict(conversation, DETAIL_CHECK=Mock(), GIFT=Mock(), NoOpportunitiesRemain=no_attempts, logger=Mock()):
            with self.assertRaises(no_attempts):
                conversation['get_next_target'](script)
        script.device.click_xy.assert_not_called()

    def test_special_button_needs_both_enabled_fill_and_correct_label(self):
        script = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'),
                                 device=SimpleNamespace(image=MagicMock()))
        for color, text, quick, expected in [((145, 145, 145), '进入战斗', False, False),
                                            ((240, 70, 20), '进入战斗', False, True),
                                            ((240, 70, 20), '模拟战战', False, False),
                                            ((245, 160, 90), '快速战斗', True, True)]:
            ocr = Mock(return_value=SimpleNamespace(ocr=lambda _: {'text': text}))
            with self.subTest(color=color, text=text), patch.dict(interception, Ocr=ocr,
                    np=SimpleNamespace(mean=lambda *args, **kwargs: color)):
                self.assertEqual(interception['special_button_enabled'](script, quick), expected)
            if color[0] == color[1]:
                ocr.assert_not_called()


class RookieOutcomeTests(unittest.TestCase):
    def test_normal_return_requires_observed_exhaustion_or_completed_quota(self):
        for board, free, quota, expected in ((True, False, 5, 'success'),
                                             (True, True, 0, 'success'),
                                             (False, False, 5, 'unknown')):
            outcomes = []
            ns = methods('module/rookie_arena/rookie_arena.py', {'run'},
                         dict(report=lambda *args, **kw: outcomes.append((args, kw)), logger=Mock(),
                              page_arena='arena', ROOKIE_ARENA_CHECK='board',
                              RookieArenaIsUnavailable=type('Unavailable', (Exception,), {}),
                              RookieArenaTimesReached=type('Reached', (Exception,), {})))
            script = SimpleNamespace(config=SimpleNamespace(RookieArena_Times=5, task_delay=Mock()),
                                     device=Mock(), ui_ensure=Mock(), ensure_into_rookie_arena=Mock(),
                                     appear=Mock(return_value=board), free_opportunity_remain=free, battle_times=quota)
            with self.subTest(board=board, free=free, quota=quota):
                ns['run'](script)
                self.assertEqual(outcomes[-1][0][0], expected)
                script.config.task_delay.assert_called_once()


class LobbyTimer:
    """Minimal Timer: reached once started, so a stable lobby frame triggers immediately."""
    def __init__(self, *args, **kwargs):
        self.on = False

    def start(self):
        self.on = True
        return self

    def started(self):
        return self.on

    def clear(self):
        self.on = False

    def reset(self):
        return self

    def reached(self):
        return self.on


class RookieEntryTests(unittest.TestCase):
    def entry(self, pages):
        failed = type('EntryFailed', (Exception,), {})
        ns = methods('module/rookie_arena/rookie_arena.py', {'ensure_into_rookie_arena'},
                     dict(Timer=LobbyTimer, logger=Mock(), NEXT_SEASON='season', MAIN_CHECK='lobby',
                          ARENA_GOTO_ROOKIE_ARENA='entry', ROOKIE_ARENA_CHECK='board', CONNECT_TO_SYSTEM='connect',
                          ROOKIE_ARENA_REENTRY=2, RookieArenaEntryFailed=failed, page_arena='arena',
                          RookieArenaIsUnavailable=type('Unavailable', (Exception,), {})))
        frames = iter(pages)
        state = SimpleNamespace(page=next(frames))

        def screenshot():
            state.page = next(frames)
        script = SimpleNamespace(device=SimpleNamespace(screenshot=screenshot), ui_ensure=Mock(),
                                 appear=lambda button, **kw: button == state.page,
                                 appear_then_click=lambda button, **kw: False,
                                 free_opportunity_remain=False, start_competition=Mock())
        return ns['ensure_into_rookie_arena'], script, failed

    def test_lobby_after_entry_click_reenters_arena_then_continues(self):
        """2026-10-22: the entry click dropped back to the lobby and the loop spun until the 900 s limit."""
        run, script, _ = self.entry(['lobby', 'board', 'board', 'board'])
        run(script)
        script.ui_ensure.assert_called_once_with('arena')

    def test_repeated_lobby_raises_after_reentry_limit(self):
        run, script, failed = self.entry(['lobby']*10)
        with self.assertRaises(failed):
            run(script)
        self.assertEqual(script.ui_ensure.call_count, 2)


class NotificationCheckTests(unittest.TestCase):
    def test_disabled_task_and_all_retries_never_send_a_message(self):
        for enabled, retry, expected in ((False, False, 'skipped'), (False, True, 'skipped'),
                                          (True, True, 'unknown')):
            outcomes, send = [], Mock()
            ns = methods('module/notify/notify.py', {'run'},
                         dict(report=lambda *args, **kw: outcomes.append((args, kw)), logger=Mock(), handle_notify=send))
            config = SimpleNamespace(Notification_WhenDailyTaskCompleted=enabled, execution_retry=retry, task_delay=Mock())
            ns['run'](SimpleNamespace(config=config))
            with self.subTest(enabled=enabled, retry=retry):
                send.assert_not_called()
                self.assertEqual(outcomes[-1][0][0], expected)


class Timer:
    def __init__(self, limit, **kwargs):
        self.limit = limit

    def start(self):
        return self

    def reset(self):
        return self

    def reached(self):
        return self.limit == 3


class SpecialRunner:
    run_special = interception['run_special']

    def __init__(self, remaining=3, quick_only=False, enabled=True):
        self.remaining, self.state, self.frames = remaining, 'panel', 0
        self.actions, self.results = [], []
        self.enabled = enabled
        self.config = SimpleNamespace(Client_Language='zh-CN', Interception_QuickBattleOnly=quick_only)
        self.device = SimpleNamespace(image=SimpleNamespace(shape=(1280, 720)),
                                      screenshot=self.screenshot, click_xy=self.click)

    def screenshot(self):
        self.frames += 1
        if self.frames > 40:
            raise AssertionError('Special interception loop did not finish')
        if self.state == 'fight':
            self.state = 'result'

    def appear(self, button, **kwargs):
        return button == self.state

    def appear_then_click(self, button, **kwargs):
        if button == 'result' and self.state == 'result':
            self.state = 'panel'
            return True
        return False

    def special_opportunities(self):
        return self.remaining

    def special_button_enabled(self, quick):
        return self.enabled and (bool(self.actions) if quick else True)

    def click(self, x, y):
        self.actions.append('quick' if y == 1105 else 'battle')
        self.remaining -= 1
        self.state = 'fight'


class SpecialFlowTests(unittest.TestCase):
    def run_script(self, **kwargs):
        script = SpecialRunner(**kwargs)
        with patch.dict(interception, Timer=Timer, logger=Mock(), END_FIGHTING='result',
                        AUTO_SHOOT='shoot', AUTO_BURST='burst', SPECIAL_INTERCEPTION_CHECK='panel',
                        report=lambda *args, **kw: script.results.append((args, kw))):
            script.run_special()
        return script

    def test_first_battle_then_two_quick_battles_finish_free_attempts(self):
        script = self.run_script()
        self.assertEqual(script.actions, ['battle', 'quick', 'quick'])
        self.assertEqual(script.remaining, 0)
        self.assertEqual(script.results[-1][0][0], 'success')

    def test_unreadable_counter_never_enters_battle(self):
        script = self.run_script(remaining=None)
        self.assertEqual(script.actions, [])
        self.assertEqual(script.results[-1][0][0], 'unknown')

    def test_disabled_actions_are_not_guessed(self):
        script = self.run_script(enabled=False)
        self.assertEqual(script.actions, [])
        self.assertEqual(script.results[-1][0][0], 'unknown')

    def test_quick_only_does_not_start_a_first_battle(self):
        script = self.run_script(quick_only=True)
        self.assertEqual(script.actions, [])
        self.assertEqual(script.results[-1][0][0], 'blocked')

    def test_zero_attempts_stop_without_any_click(self):
        script = self.run_script(remaining=0)
        self.assertEqual(script.actions, [])
        self.assertEqual(script.results[-1][0][0], 'success')


if __name__ == '__main__':
    unittest.main()
