"""Modules report observed outcomes; returning normally alone never implies success."""

import hashlib
import json
import logging
import re
from collections import deque
from contextvars import ContextVar


_current = ContextVar('execution_attempt', default=None)
_active_step = ContextVar('execution_step', default=None)


def report(status, summary, reason='', retryable=False):
    current = _current.get()
    if current is not None:
        current.report(status, summary, reason, retryable)


def plan_steps(items):
    current = _current.get()
    if current is None:
        return
    for key, label in items:
        current.steps.setdefault(key, dict(label=label, status='pending', summary='本轮尚未执行',
                                           reason='', retryable=False))
    current.checkpoint()


def step(key, label, function, *args, verified=False, **kwargs):
    current = _current.get()
    if current is None:
        return function(*args, **kwargs)
    previous = current.steps.get(key)
    if previous and previous['status'] in ('success', 'skipped'):
        from module.logger import logger
        previous['reused'] = True
        logger.info(f'Execution: keep completed step {key}')
        current.checkpoint()
        return None
    current.steps[key] = {'label': label, 'status': 'running', 'summary': '执行中', 'retryable': False, 'reason': ''}
    token = _active_step.set(key)
    current.checkpoint()
    current._step_outcomes[key] = []
    try:
        value = function(*args, **kwargs)
        if not current._step_outcomes[key]:
            report('success' if verified else 'unknown', '已完成检查' if verified else '步骤已结束，结果未确认')
        return value
    except BaseException as exc:
        outcomes = current._step_outcomes[key]
        if not any(item['status'] in ('failed', 'partial', 'blocked', 'unknown') for item in outcomes):
            status = {'EventSelectError': 'blocked', 'EventUnavailableError': 'skipped',
                      'ChallengeNotFoundError': 'unknown', 'TaskEnd': 'unknown',
                      'SystemExit': 'unknown', 'KeyboardInterrupt': 'unknown'}.get(type(exc).__name__, 'failed')
            report(status, '步骤未完成', type(exc).__name__, retryable=status in ('failed', 'unknown'))
        raise
    finally:
        current.steps[key].update(aggregate(current._step_outcomes.pop(key)))
        _active_step.reset(token)
        current.checkpoint()


def fingerprint(config, command):
    # Only a hash is retained; task configuration may contain credentials.
    data = {name: {key: value for key, value in task.items() if key not in ('Scheduler', 'Storage')}
            for name, task in config.data.items() if isinstance(task, dict)}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def aggregate(outcomes):
    items = list(outcomes)
    if not items:
        return {'status': 'unknown', 'summary': '任务已结束，尚未提供可确认的完成结果',
                'reason': '请查看详情；不能仅凭正常返回判定成功', 'retryable': False}
    attention = [item for item in items if item['status'] not in ('success', 'skipped')]
    if not attention:
        status = 'success' if any(item['status'] == 'success' for item in items) else 'skipped'
        return {'status': status, 'summary': '；'.join(dict.fromkeys(item['summary'] for item in items)),
                'reason': '', 'retryable': False}
    failed = next((item for item in attention if item['status'] == 'failed'), attention[0])
    status = 'partial' if any(item['status'] == 'success' for item in items) else failed['status']
    if status in ('pending', 'running'):
        status = 'unknown'
    return {'status': status, 'summary': '；'.join(dict.fromkeys(item['summary'] for item in attention)),
            'reason': failed.get('reason', ''), 'retryable': any(item.get('retryable') for item in attention)}


class Attempt(logging.Handler):
    def __init__(self, store, run_id, command, config):
        super().__init__(logging.INFO)
        self.store = store
        self.id, self.steps = store.begin_attempt(run_id, command, fingerprint(config, command))
        self.outcomes = []
        self._step_outcomes = {}
        self.lines = deque(maxlen=80)
        self.token = None

    def emit(self, record):
        if _current.get() is not self:
            return
        message = record.getMessage()
        if re.search(r'cookie|authorization|password|access.token|refresh.token|secret', message, re.I):
            message = '[已隐藏可能包含凭据的日志]'
        self.lines.append(f'{record.levelname}: {message[:1000]}')

    def enter(self):
        from module.logger import logger
        self.token = _current.set(self)
        logger.addHandler(self)
        return self

    def report(self, status, summary, reason='', retryable=False):
        if status not in ('success', 'partial', 'failed', 'blocked', 'skipped', 'unknown'):
            raise ValueError('Invalid task result')
        item = dict(status=status, summary=summary, reason=reason, retryable=bool(retryable))
        key = _active_step.get()
        if key:
            observations = self._step_outcomes.setdefault(key, [])
            observations.append(item)
            self.steps[key].update(aggregate(observations))
            # A checkpoint is not completion. Killing a compound step must never make it resumable as success.
            self.steps[key]['status'] = 'running'
        else:
            self.outcomes.append(item)
        self.checkpoint()

    def checkpoint(self):
        self.store.save_attempt(self.id, 'running', '执行中', steps=self.steps, logs=list(self.lines), finished=False)

    def finish(self, success=None, error=None, screenshot=None):
        from module.logger import logger
        try:
            if self.outcomes and all(item['status'] == 'skipped' for item in self.outcomes):
                for item in self.steps.values():
                    if item['status'] == 'pending':
                        item.update(status='skipped', summary='本轮无需执行')
            outcomes = self.outcomes + list(self.steps.values())
            result = aggregate(outcomes)
            if error is not None:
                if not (isinstance(error, SystemExit) and result['status'] in ('blocked', 'partial', 'failed')):
                    result = dict(status='unknown' if isinstance(error, (SystemExit, KeyboardInterrupt)) else 'failed',
                                  summary='执行中断，未能确认完成' if isinstance(error, SystemExit) else '任务异常退出',
                                  reason=type(error).__name__, retryable=False)
            elif success is False and result['status'] not in ('failed', 'partial', 'blocked'):
                result = dict(status='failed', summary='任务未完成，需要恢复游戏后重试',
                              reason='调度器已请求恢复游戏', retryable=True)
            self.store.save_attempt(self.id, **result, steps=self.steps, logs=list(self.lines), screenshot=screenshot)
        finally:
            logger.removeHandler(self)
            if self.token is not None:
                _current.reset(self.token)
