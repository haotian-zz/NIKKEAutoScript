import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from module.execution.store import ExecutionStore
from module.webui.api import routes_executions as routes


class ExecutionApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ExecutionStore(Path(self.temp.name) / 'reports.sqlite3')
        self.config = {key: {'Scheduler': {'Enable': True}} for key in ('Reward', 'Mailbox')}
        self.manager = SimpleNamespace(alive=True, start=Mock())
        self.patches = [patch.object(routes, 'ExecutionStore', return_value=self.store),
                        patch.object(routes, 'validate_instance'),
                        patch.object(routes, 'read_file', side_effect=lambda _: self.config),
                        patch.object(routes, 'get_config_mod', return_value='nkas'),
                        patch.object(routes.ProcessManager, 'get_manager', return_value=self.manager),
                        patch.object(routes.lang, 't', side_effect=lambda key: key)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.run_id = self.store.create_run('test', ['Reward', 'Mailbox'])
        for command in ('Reward', 'Mailbox'):
            attempt, _ = self.store.begin_attempt(self.run_id, command, 'hash')
            self.store.save_attempt(attempt, 'failed', '超时', retryable=True)
        self.store.finish_run(self.run_id)

    def request(self, body=None, name='test', **params):
        async def read_json():
            return body
        return SimpleNamespace(path_params=dict(name=name, run_id=self.run_id, **params),
                               query_params={}, json=read_json)

    def payload(self, tasks=None):
        return dict(tasks=tasks or ['Reward'], request_id='test-request-001',
                    revision=self.store.get_run('test', self.run_id)['revision'])

    async def test_retry_selects_only_requested_task_and_preserves_configuration(self):
        before = copy.deepcopy(self.config)
        response = await routes.retry(self.request(self.payload()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.claim_retry('test')['command'], 'Reward')
        self.assertIsNone(self.store.claim_retry('test'))
        self.assertEqual(before, self.config)
        self.manager.start.assert_not_called()

    async def test_disabled_task_is_not_force_enabled(self):
        self.config['Reward']['Scheduler']['Enable'] = False
        response = await routes.retry(self.request(self.payload()))
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.store.has_retry('test'))
        self.assertFalse(self.config['Reward']['Scheduler']['Enable'])

    async def test_invalid_body_is_rejected(self):
        response = await routes.retry(self.request(dict(tasks='Reward', request_id='valid-request', revision=0)))
        self.assertEqual(response.status_code, 400)
        self.assertFalse(self.store.has_retry('test'))

    async def test_stopped_instance_starts_only_the_retry_worker(self):
        self.manager.alive = False
        from module.webui.api import routes_instances
        with patch.object(routes_instances, '_pc_client_requires_admin', return_value=False), \
                patch.object(routes_instances, '_driver_scheme_missing_driver', return_value=False), \
                patch.object(routes_instances, '_vdd_missing_driver', return_value=False):
            response = await routes.retry(self.request(self.payload()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.start.call_args.args[0], 'execution_retry')

    async def test_report_is_scoped_to_instance(self):
        response = await routes.detail(self.request(name='other'))
        self.assertEqual(response.status_code, 404)

    async def test_unknown_report_does_not_queue_tasks(self):
        request = self.request(self.payload())
        request.path_params['run_id'] = 'unknown'
        response = await routes.retry(request)
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.store.has_retry('test'))

    async def test_list_includes_policy_and_latest_outcome(self):
        response = await routes.executions(self.request())
        data = json.loads(response.body)
        self.assertEqual(data['runs'][0]['counts']['failed'], 2)
        self.assertTrue(data['runs'][0]['tasks'][0]['can_retry'])


if __name__ == '__main__':
    unittest.main()
