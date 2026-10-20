"""Instance-scoped reports. Retries never force-enable tasks or move their schedules."""

import asyncio
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse

import module.webui.lang as lang
from module.config.utils import filepath_config, read_file
from module.execution.store import ExecutionStore, ReportConflict
from module.execution.policy import retry_policy
from module.submodule.utils import get_config_mod
from module.webui.api.deps import InstanceNotFound, validate_instance
from module.webui.process_manager import ProcessManager
from module.webui.updater import updater


def named(data):
    config = read_file(filepath_config(data['instance']))
    for task in data['tasks']:
        translated = lang.t(f'Task.{task["command"]}.name')
        task['name'] = translated if translated != f'Task.{task["command"]}.name' else task['command']
        allowed, hint = retry_policy(config, task['command'])
        task['can_retry'] = allowed and task['status'] not in ('success', 'skipped', 'running')
        task['retry_hint'] = hint
    return data


async def executions(request):
    name = request.path_params['name']
    try:
        validate_instance(name)
        limit = int(request.query_params.get('limit', 30))
        offset = int(request.query_params.get('offset', 0))
        rows = await asyncio.to_thread(ExecutionStore().list_runs, name, limit, offset)
        return JSONResponse({'runs': [named(row) for row in rows]})
    except InstanceNotFound as exc:
        return JSONResponse({'message': str(exc)}, status_code=404)
    except ValueError:
        return JSONResponse({'message': '分页参数无效'}, status_code=400)


async def detail(request):
    name, run_id = request.path_params['name'], request.path_params['run_id']
    try:
        validate_instance(name)
        row = await asyncio.to_thread(ExecutionStore().get_run, name, run_id)
        return JSONResponse(named(row)) if row else JSONResponse({'message': '执行记录不存在'}, status_code=404)
    except InstanceNotFound as exc:
        return JSONResponse({'message': str(exc)}, status_code=404)


async def retry(request):
    name, run_id = request.path_params['name'], request.path_params['run_id']
    try:
        validate_instance(name)
        if get_config_mod(name) != 'nkas':
            raise ReportConflict('此实例不支持任务重试')
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse({'message': '重试参数无效'}, status_code=400)
        commands = body.get('tasks')
        request_id = body.get('request_id')
        if (not isinstance(commands, list) or not all(isinstance(task, str) for task in commands)
                or not isinstance(request_id, str) or not 8 <= len(request_id) <= 128
                or not isinstance(body.get('revision'), int)):
            return JSONResponse({'message': '重试参数无效'}, status_code=400)
        config = read_file(filepath_config(name))
        for command in commands:
            allowed, hint = retry_policy(config, command)
            if not allowed:
                raise ReportConflict(hint)
        manager = ProcessManager.get_manager(name)
        if not manager.alive:
            from module.webui.api.routes_instances import (
                _pc_client_requires_admin, _driver_scheme_missing_driver, _vdd_missing_driver,
            )
            if _pc_client_requires_admin(name):
                raise ReportConflict('PC 客户端需要以管理员权限运行 NKAS')
            if _driver_scheme_missing_driver(name):
                raise ReportConflict('虚拟鼠标驱动未就绪，请先修复驱动')
            if _vdd_missing_driver(name):
                raise ReportConflict('虚拟屏幕驱动未就绪，请先修复驱动或关闭自动管理')
        result_id, created = ExecutionStore().enqueue(
            name, run_id, commands, request_id, body['revision'], body.get('new_day') is True,
        )
        if not manager.alive:
            manager.start('execution_retry', ev=updater.event)
        return JSONResponse({'run_id': result_id, 'queued': created, 'message': '已加入重试队列'})
    except InstanceNotFound as exc:
        return JSONResponse({'message': str(exc)}, status_code=404)
    except ReportConflict as exc:
        return JSONResponse({'message': str(exc)}, status_code=409)
    except (ValueError, TypeError):
        return JSONResponse({'message': '重试参数无效'}, status_code=400)


async def screenshot(request):
    name = request.path_params['name']
    try:
        validate_instance(name)
        store = ExecutionStore()
        value = store.screenshot(name, request.path_params['attempt_id'])
        path = Path(value).resolve() if value else None
        root = (store.path.parent / 'screenshots').resolve()
        if path and path.is_relative_to(root) and path.is_file():
            return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'private, no-store'})
        return JSONResponse({'message': '截图不存在'}, status_code=404)
    except InstanceNotFound as exc:
        return JSONResponse({'message': str(exc)}, status_code=404)
