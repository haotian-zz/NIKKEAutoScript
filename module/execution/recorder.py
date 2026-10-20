"""Execution lifecycle hooks kept separate from game and scheduling logic."""

import os

from module.execution.context import Attempt
from module.execution.store import ExecutionStore, game_day


class ExecutionRecorder:
    def __init__(self, instance):
        self.instance = instance
        self.store = ExecutionStore()
        self.run_id = None
        self.job = None

    def retry(self, config):
        if self.job is None:
            self.job = self.store.claim_retry(self.instance)
        if self.job:
            from module.execution.policy import retry_policy
            allowed, hint = retry_policy(config.data, self.job['command'])
            if not allowed:
                attempt = self.begin(self.job['command'], config)
                attempt.report('blocked', '重试前检查未通过', hint)
                attempt.finish(True)
                self.end_retry()
                return None
            config.init_task(self.job['command'])
            config.execution_retry = True
            return self.job['command']
        return None

    def begin(self, command, config, single=False):
        if self.job:
            run_id = self.job['run_id']
        else:
            if self.run_id and self.store.get_run(self.instance, self.run_id, False)['game_day'] != game_day():
                self.idle()
            if not self.run_id:
                if not single:
                    # A timer wake-up can leave the pre-wait (empty) queue cached in config.
                    config.get_next_task()
                pending = [task.command for task in config.pending_task] if not single else []
                self.run_id = self.store.create_run(self.instance, [command, *pending],
                                                    source='manual' if single else 'schedule')
            run_id = self.run_id
        return Attempt(self.store, run_id, command, config).enter()

    def end_retry(self):
        if self.job:
            self.store.finish_retry(self.job['id'])
            self.job = None

    def idle(self):
        self.store.finish_run(self.run_id)
        self.run_id = None

    def stop(self):
        self.store.interrupt(self.instance, worker_pid=os.getpid())
        self.run_id = None

    def screenshot(self, attempt, device):
        if device is None or getattr(device, 'image', None) is None:
            return None
        try:
            from PIL import Image
            path = self.store.path.parent / 'screenshots' / f'{attempt.id}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(device.image).save(path, quality=80)
            return str(path.resolve())
        except (OSError, ValueError, TypeError):
            return None
