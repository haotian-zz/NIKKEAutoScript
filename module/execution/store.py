"""SQLite is shared by the web process and automation workers, never by the browser."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


STATUSES = {'success', 'partial', 'failed', 'blocked', 'skipped', 'unknown', 'pending', 'running'}
TERMINAL = STATUSES - {'pending', 'running'}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def game_day(now=None):
    # NKAS currently schedules all supported clients against UTC+8, with a 04:00 reset.
    now = now or datetime.now(timezone.utc)
    return (now.astimezone(timezone(timedelta(hours=8))) - timedelta(hours=4)).date().isoformat()


class ReportConflict(ValueError):
    pass


class ExecutionStore:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get('NKAS_EXECUTION_DB', './data/executions/reports.sqlite3'))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, instance TEXT NOT NULL, game_day TEXT NOT NULL,
                    started_at TEXT NOT NULL, finished_at TEXT, state TEXT NOT NULL,
                    source TEXT NOT NULL, parent_id TEXT, revision INTEGER NOT NULL DEFAULT 0,
                    worker_pid INTEGER
                );
                CREATE INDEX IF NOT EXISTS runs_instance ON runs(instance, started_at DESC);
                CREATE TABLE IF NOT EXISTS tasks (
                    run_id TEXT NOT NULL, command TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    summary TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
                    retryable INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(run_id, command)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, command TEXT NOT NULL,
                    number INTEGER NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
                    status TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
                    retryable INTEGER NOT NULL DEFAULT 0, fingerprint TEXT NOT NULL,
                    steps TEXT NOT NULL DEFAULT '{}', logs TEXT NOT NULL DEFAULT '[]', screenshot TEXT,
                    UNIQUE(run_id, command, number)
                );
                CREATE TABLE IF NOT EXISTS retries (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, command TEXT NOT NULL,
                    instance TEXT NOT NULL, state TEXT NOT NULL, created_at TEXT NOT NULL,
                    request_id TEXT NOT NULL, worker_pid INTEGER
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_retry ON retries(instance, command)
                    WHERE state IN ('queued', 'running');
            ''')

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_run(self, instance, commands, source='schedule', parent_id=None):
        run_id = uuid4().hex
        with self.connect(write=True) as db:
            db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)', (
                run_id, instance, game_day(), timestamp(), None, 'running', source, parent_id, 0, os.getpid(),
            ))
            db.executemany('INSERT INTO tasks(run_id,command) VALUES (?,?)',
                           [(run_id, command) for command in dict.fromkeys(commands)])
        return run_id

    def finish_run(self, run_id, interrupted=False):
        if not run_id:
            return
        with self.connect(write=True) as db:
            db.execute('UPDATE runs SET state=?, finished_at=?, revision=revision+1 WHERE id=? AND state=?',
                       ('interrupted' if interrupted else 'finished', timestamp(), run_id, 'running'))

    def begin_attempt(self, run_id, command, fingerprint):
        attempt_id = uuid4().hex
        with self.connect(write=True) as db:
            db.execute('INSERT OR IGNORE INTO tasks(run_id,command) VALUES (?,?)', (run_id, command))
            previous = db.execute('SELECT * FROM attempts WHERE run_id=? AND command=? ORDER BY number DESC LIMIT 1',
                                  (run_id, command)).fetchone()
            number = previous['number'] + 1 if previous else 1
            steps = json.loads(previous['steps']) if previous and previous['fingerprint'] == fingerprint else {}
            db.execute('INSERT INTO attempts(id,run_id,command,number,started_at,status,fingerprint,steps) '
                       'VALUES (?,?,?,?,?,?,?,?)',
                       (attempt_id, run_id, command, number, timestamp(), 'running', fingerprint, json.dumps(steps)))
            db.execute('UPDATE tasks SET status=?,attempts=? WHERE run_id=? AND command=?',
                       ('running', number, run_id, command))
            db.execute('UPDATE runs SET revision=revision+1 WHERE id=?', (run_id,))
        return attempt_id, steps

    def save_attempt(self, attempt_id, status, summary, reason='', retryable=False, steps=None, logs=None,
                     screenshot=None, finished=True):
        if status not in STATUSES:
            raise ValueError('Invalid outcome')
        with self.connect(write=True) as db:
            row = db.execute('SELECT * FROM attempts WHERE id=?', (attempt_id,)).fetchone()
            if not row or row['status'] != 'running':
                return
            db.execute('UPDATE attempts SET status=?,summary=?,reason=?,retryable=?,steps=?,logs=?,screenshot=?, '
                       'finished_at=? WHERE id=?', (
                           status, summary, reason, int(retryable), json.dumps(steps or {}, ensure_ascii=False),
                           json.dumps(logs or [], ensure_ascii=False), screenshot,
                           timestamp() if finished else None, attempt_id,
                       ))
            db.execute('UPDATE tasks SET status=?,summary=?,reason=?,retryable=? WHERE run_id=? AND command=?',
                       (status, summary, reason, int(retryable), row['run_id'], row['command']))
            db.execute('UPDATE runs SET revision=revision+1 WHERE id=?', (row['run_id'],))

    def get_run(self, instance, run_id, details=True):
        with self.connect() as db:
            row = db.execute('SELECT * FROM runs WHERE id=? AND instance=?', (run_id, instance)).fetchone()
            return self._run_data(db, row, details) if row else None

    def list_runs(self, instance, limit=30, offset=0):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM runs WHERE instance=? ORDER BY started_at DESC,rowid DESC LIMIT ? OFFSET ?',
                              (instance, min(max(limit, 1), 100), max(offset, 0))).fetchall()
            return [self._run_data(db, row, False) for row in rows]

    def _run_data(self, db, row, details):
        data = dict(row)
        data.pop('worker_pid', None)
        tasks = [dict(item) for item in db.execute('SELECT * FROM tasks WHERE run_id=? ORDER BY rowid', (row['id'],))]
        active = {item['command']: item['state'] for item in db.execute(
            'SELECT command,state FROM retries WHERE run_id=? AND state IN (?,?)', (row['id'], 'queued', 'running'))}
        counts = {status: 0 for status in STATUSES}
        for task in tasks:
            task['retryable'] = bool(task['retryable'])
            task['retry_state'] = active.get(task['command'])
            counts[task['status']] += 1
            if details:
                attempts = db.execute('SELECT * FROM attempts WHERE run_id=? AND command=? ORDER BY number DESC',
                                      (row['id'], task['command'])).fetchall()
                task['history'] = []
                for item in attempts:
                    attempt = dict(item)
                    attempt['steps'] = json.loads(attempt['steps'])
                    attempt['logs'] = json.loads(attempt['logs'])
                    attempt['has_screenshot'] = bool(attempt.pop('screenshot'))
                    attempt.pop('fingerprint', None)
                    task['history'].append(attempt)
        data.update(counts=counts, tasks=tasks, retrying=bool(active), stale=row['game_day'] != game_day())
        duration = db.execute('SELECT SUM(MAX(0, (julianday(COALESCE(finished_at,?)) - '
                              'julianday(started_at))*86400)) FROM attempts WHERE run_id=?',
                              (timestamp(), row['id'])).fetchone()[0]
        data['duration_seconds'] = round(duration or 0)
        return data

    def enqueue(self, instance, run_id, commands, request_id, revision, new_day=False):
        with self.connect(write=True) as db:
            duplicate = db.execute('SELECT run_id FROM retries WHERE instance=? AND request_id=? LIMIT 1',
                                   (instance, request_id)).fetchone()
            if duplicate:
                return duplicate['run_id'], False
            run = db.execute('SELECT * FROM runs WHERE id=? AND instance=?', (run_id, instance)).fetchone()
            if not run:
                raise ReportConflict('执行记录不存在')
            if run['revision'] != revision:
                raise ReportConflict('执行结果已更新，请刷新后重试')
            if run['state'] == 'running':
                raise ReportConflict('本轮还在执行，请结束后重试')
            stale = run['game_day'] != game_day()
            if stale and not new_day:
                raise ReportConflict('已跨过游戏每日刷新时间，请按当前游戏日重新执行')
            commands = list(dict.fromkeys(commands))
            if not commands or len(commands) > 100:
                raise ReportConflict('请选择要重试的任务')
            for command in commands:
                task = db.execute('SELECT * FROM tasks WHERE run_id=? AND command=?', (run_id, command)).fetchone()
                if not task or task['status'] in ('success', 'skipped', 'running'):
                    raise ReportConflict('所选任务已完成或当前不可重试')
                if db.execute('SELECT 1 FROM retries WHERE instance=? AND command=? AND state IN (?,?)',
                              (instance, command, 'queued', 'running')).fetchone():
                    raise ReportConflict('所选任务已经在重试队列中')
            if stale:
                old_id, run_id = run_id, uuid4().hex
                db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)', (
                    run_id, instance, game_day(), timestamp(), None, 'finished', 'retry', old_id, 0, None,
                ))
                db.executemany('INSERT INTO tasks(run_id,command) VALUES (?,?)', [(run_id, task) for task in commands])
            db.executemany('INSERT INTO retries VALUES (?,?,?,?,?,?,?,?)', [
                (uuid4().hex, run_id, task, instance, 'queued', timestamp(), request_id, None)
                for task in commands
            ])
            db.execute('UPDATE runs SET revision=revision+1 WHERE id=?', (run_id,))
        return run_id, True

    def has_retry(self, instance):
        with self.connect() as db:
            return bool(db.execute('SELECT 1 FROM retries WHERE instance=? AND state=? LIMIT 1',
                                   (instance, 'queued')).fetchone())

    def queue(self, instance):
        with self.connect() as db:
            current = db.execute('SELECT a.command FROM attempts a JOIN runs r ON a.run_id=r.id '
                                 'WHERE r.instance=? AND a.status="running" ORDER BY a.rowid DESC LIMIT 1',
                                 (instance,)).fetchone()
            queued = db.execute('SELECT command,created_at FROM retries WHERE instance=? AND state="queued" '
                                'ORDER BY rowid', (instance,)).fetchall()
            return (current['command'] if current else None), [dict(row) for row in queued]

    def claim_retry(self, instance):
        with self.connect(write=True) as db:
            db.execute('UPDATE retries SET state="cancelled" WHERE instance=? AND state="queued" '
                       'AND run_id IN (SELECT id FROM runs WHERE game_day != ?)', (instance, game_day()))
            row = db.execute('SELECT * FROM retries WHERE instance=? AND state=? ORDER BY rowid LIMIT 1',
                             (instance, 'queued')).fetchone()
            if not row:
                return None
            task = db.execute('SELECT status FROM tasks WHERE run_id=? AND command=?',
                              (row['run_id'], row['command'])).fetchone()
            if task and task['status'] == 'success':
                db.execute('UPDATE retries SET state="done" WHERE id=?', (row['id'],))
                return None
            db.execute('UPDATE retries SET state=?,worker_pid=? WHERE id=?', ('running', os.getpid(), row['id']))
            return dict(row)

    def finish_retry(self, retry_id):
        with self.connect(write=True) as db:
            db.execute('UPDATE retries SET state=? WHERE id=?', ('done', retry_id))
            db.execute('UPDATE runs SET finished_at=?,revision=revision+1 WHERE id='
                       '(SELECT run_id FROM retries WHERE id=?)', (timestamp(), retry_id))
            db.execute('UPDATE runs SET state="finished" WHERE id='
                       '(SELECT run_id FROM retries WHERE id=?) AND state="interrupted" '
                       'AND NOT EXISTS (SELECT 1 FROM tasks WHERE run_id=runs.id '
                       'AND status NOT IN ("success","skipped")) '
                       'AND NOT EXISTS (SELECT 1 FROM retries WHERE run_id=runs.id '
                       'AND state IN ("queued","running"))', (retry_id,))

    def interrupt(self, instance, summary='执行已停止；进行中的操作结果尚未确认', worker_pid=None,
                  include_queued=True):
        with self.connect(write=True) as db:
            where = ' AND r.worker_pid=?' if worker_pid is not None else ''
            params = (instance, worker_pid) if worker_pid is not None else (instance,)
            runs = db.execute('SELECT r.id FROM runs r WHERE r.instance=? AND r.state="running"' + where,
                              params).fetchall()
            for run in runs:
                db.execute('UPDATE runs SET state=?,finished_at=?,revision=revision+1 WHERE id=?',
                           ('interrupted', timestamp(), run['id']))
            # Retry attempts belong to a finished report, so they are selected through the owning job as well.
            states = '("queued","running")' if include_queued else '("running")'
            jobs = db.execute('SELECT * FROM retries r WHERE r.instance=? AND r.state IN ' + states + where,
                              params).fetchall()
            affected = {run['id'] for run in runs} | {job['run_id'] for job in jobs}
            for run_id in affected:
                attempts = db.execute('SELECT id,steps FROM attempts WHERE run_id=? AND status="running"',
                                      (run_id,)).fetchall()
                for attempt in attempts:
                    steps = json.loads(attempt['steps'])
                    for item in steps.values():
                        if item['status'] == 'running':
                            item.update(status='unknown', summary=summary, retryable=False)
                    db.execute('UPDATE attempts SET steps=? WHERE id=?',
                               (json.dumps(steps, ensure_ascii=False), attempt['id']))
                db.execute('UPDATE attempts SET status=?,summary=?,retryable=0,finished_at=? '
                           'WHERE run_id=? AND status=?', ('unknown', summary, timestamp(), run_id, 'running'))
                db.execute('UPDATE tasks SET status=?,summary=?,retryable=0 WHERE run_id=? AND status=?',
                           ('unknown', summary, run_id, 'running'))
                db.execute('UPDATE runs SET revision=revision+1 WHERE id=?', (run_id,))
            for job in jobs:
                db.execute('UPDATE retries SET state=? WHERE id=?', ('cancelled', job['id']))

    def screenshot(self, instance, attempt_id):
        with self.connect() as db:
            row = db.execute('SELECT a.screenshot FROM attempts a JOIN runs r ON a.run_id=r.id '
                             'WHERE a.id=? AND r.instance=?', (attempt_id, instance)).fetchone()
            return row['screenshot'] if row else None
