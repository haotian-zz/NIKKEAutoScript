import logging
import ast
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from module.execution.context import Attempt, aggregate, plan_steps, report, step
from module.execution.policy import retry_policy
from module.execution.store import ExecutionStore, ReportConflict, game_day


class ExecutionReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ExecutionStore(Path(self.temp.name) / 'reports.sqlite3')
        self.config = SimpleNamespace(data={'Reward': {'Scheduler': {'Enable': True}, 'Reward': {'Enabled': True}}})
        self.logger_patch = patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=logging.getLogger('test'))})
        self.logger_patch.start()
        self.addCleanup(self.logger_patch.stop)

    def failed_run(self, commands=('Reward',)):
        run_id = self.store.create_run('test', commands)
        for command in commands:
            attempt = Attempt(self.store, run_id, command, self.config).enter()
            report('failed', '入口超时', retryable=True)
            attempt.finish(True)
        self.store.finish_run(run_id)
        return self.store.get_run('test', run_id)

    def enqueue(self, run, commands=('Reward',), request_id='request-0001', **kwargs):
        return self.store.enqueue('test', run['id'], list(commands), request_id, run['revision'], **kwargs)

    def test_retry_updates_original_but_keeps_attempts(self):
        run = self.failed_run()
        self.enqueue(run)
        job = self.store.claim_retry('test')
        attempt = Attempt(self.store, job['run_id'], job['command'], self.config).enter()
        report('success', '奖励已领取')
        attempt.finish(True)
        self.store.finish_retry(job['id'])
        final = self.store.get_run('test', run['id'])
        self.assertEqual(final['counts']['failed'], 0)
        self.assertEqual(final['counts']['success'], 1)
        self.assertEqual([item['status'] for item in final['tasks'][0]['history']], ['success', 'failed'])
        self.assertFalse(final['retrying'])

    def test_double_submission_is_idempotent(self):
        run = self.failed_run()
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda _: self.enqueue(run), range(2)))
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertIsNotNone(self.store.claim_retry('test'))
        self.assertIsNone(self.store.claim_retry('test'))

    def test_conflicting_or_stale_selection_rejected(self):
        run = self.failed_run()
        self.enqueue(run)
        with self.assertRaises(ReportConflict):
            self.enqueue(run, request_id='request-0002')
        self.assertIsNone(self.store.get_run('another-instance', run['id']))

    def test_cross_day_creates_new_report(self):
        run = self.failed_run()
        with self.store.connect(write=True) as db:
            db.execute('UPDATE runs SET game_day="2020-01-01" WHERE id=?', (run['id'],))
        with self.assertRaises(ReportConflict):
            self.enqueue(run)
        new_id, _ = self.enqueue(run, new_day=True)
        self.assertNotEqual(new_id, run['id'])
        self.assertEqual(self.store.get_run('test', new_id)['parent_id'], run['id'])
        self.assertEqual(self.store.get_run('test', run['id'])['tasks'][0]['status'], 'failed')

    def test_stop_is_unknown_and_unstarted_is_pending(self):
        run_id = self.store.create_run('test', ['Reward', 'Mailbox'])
        self.store.begin_attempt(run_id, 'Reward', 'config-hash')
        self.store.interrupt('test')
        final = self.store.get_run('test', run_id)
        self.assertEqual(final['state'], 'interrupted')
        self.assertEqual([task['status'] for task in final['tasks']], ['unknown', 'pending'])

    def test_start_recovery_preserves_newly_queued_retries(self):
        run = self.failed_run()
        self.enqueue(run)
        self.store.interrupt('test', include_queued=False)
        self.assertTrue(self.store.has_retry('test'))

    def test_resume_skips_successful_steps_and_retries_failure(self):
        run_id = self.store.create_run('test', ['Reward'])
        calls = []
        first = Attempt(self.store, run_id, 'Reward', self.config).enter()
        step('defense', '收菜', lambda: calls.append('defense'), verified=True)
        step('ranking', '排名', lambda: report('failed', '排名超时', retryable=True))
        first.finish(True)
        second = Attempt(self.store, run_id, 'Reward', self.config).enter()
        step('defense', '收菜', lambda: calls.append('duplicate'), verified=True)
        step('ranking', '排名', lambda: report('success', '排名已完成'))
        second.finish(True)
        self.assertEqual(calls, ['defense'])
        final = self.store.get_run('test', run_id)
        self.assertEqual(final['tasks'][0]['status'], 'success')
        self.assertEqual(len(final['tasks'][0]['history']), 2)

    def test_config_change_invalidates_step_resume(self):
        run_id = self.store.create_run('test', ['Reward'])
        first = Attempt(self.store, run_id, 'Reward', self.config).enter()
        step('collect', '领取', lambda: None, verified=True)
        first.finish(True)
        self.config.data['Reward']['Reward']['Enabled'] = False
        second = Attempt(self.store, run_id, 'Reward', self.config).enter()
        self.assertEqual(second.steps, {})
        second.finish(True)

    def test_normal_return_without_evidence_is_unknown(self):
        self.assertEqual(aggregate([])['status'], 'unknown')
        run_id = self.store.create_run('test', ['Reward'])
        attempt = Attempt(self.store, run_id, 'Reward', self.config).enter()
        attempt.finish(True)
        self.assertEqual(self.store.get_run('test', run_id)['tasks'][0]['status'], 'unknown')

    def test_skip_does_not_erase_failure(self):
        result = aggregate([{'status': 'failed', 'summary': '领奖超时', 'retryable': True},
                            {'status': 'skipped', 'summary': '活动未开放'}])
        self.assertEqual(result['status'], 'failed')

    def test_scheduler_failure_keeps_specific_timeout_reason(self):
        run_id = self.store.create_run('test', ['Reward'])
        attempt = Attempt(self.store, run_id, 'Reward', self.config).enter()
        step('ranking', '排名', lambda: report('failed', '排名页面超时', '90 秒未确认进展', retryable=True))
        attempt.finish(False)
        result = self.store.get_run('test', run_id)['tasks'][0]
        self.assertEqual(result['summary'], '排名页面超时')
        self.assertEqual(result['reason'], '90 秒未确认进展')
        self.assertTrue(result['retryable'])

    def test_intermediate_success_does_not_finish_compound_step(self):
        run_id = self.store.create_run('test', ['Reward'])
        attempt = Attempt(self.store, run_id, 'Reward', self.config).enter()
        def combined():
            report('success', 'Normal 已通关')
            saved = self.store.get_run('test', run_id)['tasks'][0]['history'][0]['steps']
            self.assertEqual(saved['story']['status'], 'running')
            report('failed', 'Hard 等待超时', retryable=True)
        step('story', '活动推进', combined)
        attempt.finish(True)
        self.assertEqual(self.store.get_run('test', run_id)['tasks'][0]['status'], 'partial')

    def test_interrupt_mid_step_does_not_skip_remaining_work_on_retry(self):
        run_id = self.store.create_run('test', ['Reward'])
        first = Attempt(self.store, run_id, 'Reward', self.config).enter()
        def interrupted():
            report('success', 'Normal 已通关')
            self.store.interrupt('test')
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            step('story', '活动推进', interrupted)
        first.finish(error=KeyboardInterrupt())
        saved = self.store.get_run('test', run_id)['tasks'][0]['history'][0]['steps']
        self.assertEqual(saved['story']['status'], 'unknown')
        second = Attempt(self.store, run_id, 'Reward', self.config).enter()
        calls = []
        step('story', '活动推进', lambda: calls.append('resume'), verified=True)
        second.finish(True)
        self.assertEqual(calls, ['resume'])

    def test_daily_reset_uses_server_day(self):
        self.assertEqual(game_day(datetime(2026, 10, 20, 19, 59, tzinfo=timezone.utc)), '2026-10-20')
        self.assertEqual(game_day(datetime(2026, 10, 20, 20, 0, tzinfo=timezone.utc)), '2026-10-21')

    def test_consumptive_tasks_cannot_be_replayed_blindly(self):
        config = {name: {'Scheduler': {'Enable': True}} for name in ('DailyRecruit', 'Destruction', 'Reward')}
        self.assertFalse(retry_policy(config, 'DailyRecruit')[0])
        self.assertTrue(retry_policy(config, 'Reward')[0])
        config['Destruction']['Destruction'] = {'UseGem': True}
        self.assertFalse(retry_policy(config, 'Destruction')[0])

    def test_unstarted_steps_are_visible_and_never_marked_success(self):
        run_id = self.store.create_run('test', ['Reward'])
        attempt = Attempt(self.store, run_id, 'Reward', self.config).enter()
        plan_steps([('first', '已完成步骤'), ('second', '未执行步骤')])
        step('first', '已完成步骤', lambda: None, verified=True)
        attempt.finish(True)
        result = self.store.get_run('test', run_id)['tasks'][0]
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['history'][0]['steps']['second']['status'], 'pending')

    def test_account_prerequisite_is_not_a_technical_failure(self):
        class EventSelectError(Exception):
            pass
        run_id = self.store.create_run('test', ['Reward'])
        attempt = Attempt(self.store, run_id, 'Reward', self.config).enter()
        def locked():
            raise EventSelectError
        with self.assertRaises(EventSelectError):
            step('story', '困难剧情', locked)
        attempt.finish(True)
        self.assertEqual(self.store.get_run('test', run_id)['tasks'][0]['status'], 'blocked')

    def test_queued_retry_cannot_overwrite_previous_game_day(self):
        run = self.failed_run()
        self.enqueue(run)
        with self.store.connect(write=True) as db:
            db.execute('UPDATE runs SET game_day="2020-01-01" WHERE id=?', (run['id'],))
        self.assertIsNone(self.store.claim_retry('test'))
        self.assertFalse(self.store.has_retry('test'))
        self.assertEqual(self.store.get_run('test', run['id'])['tasks'][0]['status'], 'failed')

    def source_method(self, filename, name, namespace):
        tree = ast.parse((Path(__file__).resolve().parents[1] / filename).read_text(encoding='utf-8'))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[method], type_ignores=[]), filename, 'exec'), namespace)
        return namespace[name]

    def test_stopped_retry_loop_never_selects_other_due_tasks(self):
        from unittest.mock import Mock
        namespace = dict(logger=Mock(), datetime=datetime,
                         inflection=SimpleNamespace(underscore=str.lower), del_cached_property=Mock())
        method = self.source_method('main.py', 'loop_retries', namespace)
        execution = SimpleNamespace(job=None, retry=Mock(side_effect=['Reward', 'Mailbox']), stop=Mock(),
                                    store=SimpleNamespace(has_retry=Mock(side_effect=[True, True, False])))
        script = SimpleNamespace(config_name='test', config=SimpleNamespace(get_next=Mock()),
                                 execution=execution, stop_event=None, serial_report_due=Mock(),
                                 serial_wait_turn=Mock(return_value=True), driver_wait_turn=Mock(return_value=True),
                                 run_recorded=Mock(), driver_release=Mock())
        method(script)
        self.assertEqual([call.args[0] for call in script.run_recorded.call_args_list], ['reward', 'mailbox'])
        script.config.get_next.assert_not_called()
        script.driver_release.assert_called_once()

    def test_retry_does_not_move_normal_schedule(self):
        method = self.source_method('module/config/config.py', 'task_delay', {})
        config = SimpleNamespace(execution_retry=True, modified={'Reward.Scheduler.NextRun': 'unchanged'})
        method(config, schedule=True)
        self.assertEqual(config.modified, {'Reward.Scheduler.NextRun': 'unchanged'})

    def login_recovery_runner(self, outcomes):
        from unittest.mock import Mock
        import module.exception as exceptions
        namespace = dict(vars(exceptions), logger=Mock(), TaskEnd=type('TaskEnd', (Exception,), {}))
        method = self.source_method('main.py', 'run', namespace)
        script = SimpleNamespace(is_independent_task=Mock(return_value=False), device=Mock(),
                                 event=Mock(side_effect=outcomes), start=Mock(), config=Mock(), save_error_log=Mock())
        return method, script

    def test_selected_task_continues_after_login_recovery(self):
        from module.exception import GameStart
        method, script = self.login_recovery_runner([GameStart(), None])
        self.assertTrue(method(script, 'event'))
        self.assertEqual(script.event.call_count, 2)
        script.start.assert_called_once()
        script.config.task_call.assert_not_called()

    def test_repeated_login_recovery_is_bounded(self):
        from module.exception import GameStart
        method, script = self.login_recovery_runner([GameStart()] * 4)
        self.assertFalse(method(script, 'event'))
        self.assertEqual(script.start.call_count, 3)
        script.config.task_call.assert_called_once_with('Restart')

    def test_interrupted_report_finishes_when_all_tasks_recover(self):
        run_id = self.store.create_run('test', ['Reward'])
        self.store.begin_attempt(run_id, 'Reward', 'hash')
        self.store.interrupt('test')
        run = self.store.get_run('test', run_id)
        self.enqueue(run)
        job = self.store.claim_retry('test')
        attempt, _ = self.store.begin_attempt(run_id, 'Reward', 'hash')
        self.store.save_attempt(attempt, 'success', '领取已确认')
        self.store.finish_retry(job['id'])
        final = self.store.get_run('test', run_id)
        self.assertEqual(final['state'], 'finished')
        self.assertEqual([item['status'] for item in final['tasks'][0]['history']], ['success', 'unknown'])

    def test_recovering_one_task_does_not_hide_unstarted_tasks(self):
        run_id = self.store.create_run('test', ['Reward', 'Mailbox'])
        self.store.begin_attempt(run_id, 'Reward', 'hash')
        self.store.interrupt('test')
        self.enqueue(self.store.get_run('test', run_id))
        job = self.store.claim_retry('test')
        attempt, _ = self.store.begin_attempt(run_id, 'Reward', 'hash')
        self.store.save_attempt(attempt, 'success', '领取已确认')
        self.store.finish_retry(job['id'])
        final = self.store.get_run('test', run_id)
        self.assertEqual(final['state'], 'interrupted')
        self.assertEqual(final['counts']['pending'], 1)


if __name__ == '__main__':
    unittest.main()
