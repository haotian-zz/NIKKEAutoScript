"""Exercise the real page orchestration without importing game assets or sending input."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = ast.parse((Path(__file__).resolve().parents[1] / 'module/event/story.py').read_text())
methods = [node for node in source.body if isinstance(node, ast.ClassDef)][0].body
body = [node for node in methods if isinstance(node, ast.FunctionDef) and node.name in
        ('story', 'story_modes', 'stage_assets', 'STORY_STAGE_11', 'STORY_STAGE_12', 'find_and_sweep_stage')]
namespace = {'logger': SimpleNamespace(hr=lambda *args: None, info=lambda *args: None),
             'report': lambda *args: None, 're': __import__('re'), 'EventSelectError': ValueError}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'event_orchestration', 'exec'), namespace)


class Runner:
    story = namespace['story']
    story_modes = namespace['story_modes']
    stage_assets = namespace['stage_assets']
    STORY_STAGE_11 = namespace['STORY_STAGE_11']
    STORY_STAGE_12 = namespace['STORY_STAGE_12']

    def __init__(self, kind=2, unlocked=None, releases=None, tickets=False):
        self.event = SimpleNamespace(type=kind)
        self.config = SimpleNamespace(StoryStage_AutoPush=True, StoryStage_Sweep=True)
        self.device = SimpleNamespace(screenshot=lambda: None)
        self.unlocked = set(unlocked or ['story_1_normal'])
        self.releases = releases or {}
        self.tickets = tickets
        self.actions = []

    def back_to_event(self):
        self.actions.append(('home', None))

    def open_story_page(self, page):
        self.actions.append(('try', page))
        return page in self.unlocked

    def find_and_push_stage(self, page):
        self.actions.append(('push', page))
        self.unlocked.update(self.releases.get(page, []))
        return 'tickets' if self.tickets else 'cleared'

    def page_cleared(self, page):
        return not self.tickets

    def find_and_sweep_stage(self, page):
        self.actions.append(('sweep', page))


class EventProgressionTests(unittest.TestCase):
    def test_normal_clear_unlocks_hard_before_spending_repeat_tickets(self):
        run = Runner(releases={'story_1_normal': ['story_1_hard']})
        run.story()
        self.assertEqual([item for item in run.actions if item[0] in ('push', 'sweep')],
                         [('push', 'story_1_normal'), ('push', 'story_1_hard'), ('sweep', 'story_1_hard')])

    def test_time_gate_keeps_sweep_on_normal(self):
        run = Runner()
        run.story()
        self.assertIn(('sweep', 'story_1_normal'), run.actions)
        self.assertEqual(run.actions.count(('try', 'story_1_hard')), 2)

    def test_large_event_walks_three_pages(self):
        run = Runner(kind=1, releases={'story_1_normal': ['story_2_normal'],
                                      'story_2_normal': ['story_2_hard']})
        run.story()
        self.assertEqual([value for action, value in run.actions if action == 'push'],
                         ['story_1_normal', 'story_2_normal', 'story_2_hard'])
        self.assertIn(('sweep', 'story_2_hard'), run.actions)

    def test_no_tickets_does_not_loop_or_switch_to_locked_page(self):
        run = Runner(tickets=True)
        run.story()
        self.assertFalse(any(action == 'sweep' for action, _ in run.actions))
        self.assertEqual(run.actions.count(('try', 'story_1_hard')), 1)

    def test_already_unlocked_hard_has_priority(self):
        run = Runner(unlocked=['story_1_normal', 'story_1_hard'])
        run.story()
        self.assertNotIn(('push', 'story_1_normal'), run.actions)

    def test_penultimate_is_derived_from_assets(self):
        for count in (10, 12):
            run = Runner()
            names = {f'STORY_1_NORMAL_STAGE_{i}{suffix}': (i, suffix)
                     for i in (count - 1, count) for suffix in ('', '_CLEAR')}
            run.event_assets = SimpleNamespace(**names)
            self.assertEqual(run.STORY_STAGE_11('story_1_normal')[0], count - 1)
            self.assertEqual(run.STORY_STAGE_12('story_1_normal_clear')[0], count)


class SweepRunner:
    find_and_sweep_stage = namespace['find_and_sweep_stage']

    def __init__(self, unreadable=False):
        self.state, self.tickets, self.batches = 'map', 25, 0
        self.targets, self.results = [], []
        self.unreadable = unreadable
        self.device = SimpleNamespace(image=None, screenshot=lambda: None, sleep=lambda _: None)
        self.config = SimpleNamespace(Optimization_OcrModelType='mobile')
        self.event_assets = SimpleNamespace(STORY_STAGE_CHECK='panel')
        self.ensure_sroll_to_bottom = lambda **kwargs: None
        self.open_team = lambda: None
        self.team_up = lambda **kwargs: None
        self.stage_assets = lambda _: [(9, 'stage9', 'clear9'), (10, 'stage10', 'clear10')]
        self.STORY_PART_CHECK = lambda _: 'map'
        namespace.update(WaitGuard=lambda *args, **kwargs: SimpleNamespace(check=lambda: None, progress=lambda: None),
                         Ocr=lambda *args, **kwargs: SimpleNamespace(ocr=lambda _: {'text': '1-9 EVENT'}),
                         report=lambda *args, **kwargs: self.results.append((args, kwargs)),
                         END_FIGHTING='result', CHALLENGE_QUICKLY_DISABLE='disabled', FIGHT_CLOSE='close',
                         FIGHT_QUICKLY='quick', FIGHT_QUICKLY_CHECK='quick-panel',
                         FIGHT_QUICKLY_MAX='max', FIGHT_QUICKLY_FIGHT='fight')

    def event_tickets(self):
        return None if self.unreadable and self.batches else self.tickets

    def appear(self, button, **kwargs):
        if button == 'clear9':
            return True
        if button == 'quick-panel':
            return self.state == 'quick'
        if button == 'disabled':
            return self.state == 'panel' and self.tickets == 0
        return button == self.state

    def appear_then_click(self, button, **kwargs):
        if button == 'stage9' and self.state == 'map':
            self.targets.append(button)
            self.state = 'panel'
        elif button == 'quick' and self.state == 'panel':
            self.state = 'quick'
        elif button == 'max' and self.state == 'quick':
            pass
        elif button == 'fight' and self.state == 'quick':
            self.tickets = max(0, self.tickets - 20)
            self.batches += 1
            self.state = 'result'
        elif button == 'result' and self.state == 'result':
            self.state = 'map'
        elif button == 'close' and self.state == 'panel':
            self.state = 'map'
        else:
            return False
        return True


class EventSweepTests(unittest.TestCase):
    def test_repeat_uses_penultimate_until_all_tickets_are_consumed(self):
        run = SweepRunner()
        run.find_and_sweep_stage('story_1_hard')
        self.assertEqual(run.targets, ['stage9', 'stage9'])
        self.assertEqual(run.tickets, 0)
        self.assertIn('免费票为零', run.results[-1][0][1])

    def test_unreadable_remaining_counter_is_not_reported_as_all_spent(self):
        run = SweepRunner(unreadable=True)
        run.find_and_sweep_stage('story_1_hard')
        self.assertEqual(run.results[-1][0][0], 'unknown')
        self.assertEqual(run.batches, 1)


if __name__ == '__main__':
    unittest.main()
