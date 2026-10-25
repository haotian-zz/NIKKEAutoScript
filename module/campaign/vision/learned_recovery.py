"""可选学习型设施原型适配器；输入仍由正式会话执行，模型不持有鼠标。"""

import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid

import cv2
import numpy as np

from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.navigation import controls as goto
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.vision.objects import detect_objects
from module.campaign.vision.perception import squad_arrow
from module.logger import logger

ROOT = Path(__file__).resolve().parents[3] / 'data/vision_runtime'


def enabled(runner):
    """仅明确启用的AutoPush实例可用原型；其他实例保持既有执行路径。"""
    path = ROOT / 'prototype.json'
    if not path.exists():
        return False
    settings = json.loads(path.read_text(encoding='utf8'))
    return bool(settings.get('enabled') is True and settings.get('instance') == 'AutoPush'
                and runner.owner.config.config_name == 'AutoPush')


class Recovery:
    """跨多次设施恢复保留原图与动作证据；每场胜利前只允许有限原型动作。"""

    def __init__(self, runner):
        """绑定当前正式会话，独立spool不跨实例或账号复用。"""
        self.runner = runner
        self.spool = ROOT / 'spool'
        self.spool.mkdir(parents=True, exist_ok=True)
        self.steps = 0
        self.wins = len(runner.owner.story_wins)

    def snapshot(self, field):
        """同一个原子NPZ保存原始画面与候选；未确认小队时不生成设施落点。"""
        arrow = squad_arrow(field)
        if arrow is None:
            raise ValueError('学习型原型未确认画面内小队，未生成移动计划。')
        metadata = dict(ground=(arrow + [0., goto.ARROW_GROUND_OFFSET]).tolist(),
                        objects=detect_objects(field), captured_at=time.time())
        name = uuid.uuid4().hex + '.npz'
        temporary = self.spool / (name + '.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, field=field, metadata=json.dumps(metadata))
        temporary.replace(self.spool / name)
        return name, metadata

    def request(self, name, previous=None):
        """本机固定回环端口调用只读模型；核对回包帧哈希与客户区，超时不能继续输入。"""
        with np.load(self.spool / name, allow_pickle=False) as archive:
            digest = hashlib.sha256(archive['field'].tobytes()).hexdigest()
        payload = {'frame': name}
        if previous is not None:
            payload['previous'] = previous
        request = urllib.request.Request('http://127.0.0.1:12279/infer',
                                         json.dumps(payload).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=12 if previous else 90) as response:
            result = json.load(response)
        if result.get('schema') != 1 or result.get('source_sha256') != digest or result.get('size') != [1776, 999]:
            raise ValueError('学习型原型回包不属于本次原始帧。')
        if previous is not None:
            with np.load(self.spool / previous, allow_pickle=False) as archive:
                expected = hashlib.sha256(archive['field'].tobytes()).hexdigest()
            if result.get('previous_sha256') != expected:
                raise ValueError('学习型原型配准不属于指定的前一帧。')
        self.runner.session.check()
        return result

    def recheck(self, name, field, point, ground):
        """推理后新拍一帧复核静态几何与小队；镜头或角色变化使原计划失效。"""
        fresh = self.runner.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            raise TargetTriggered('设施规划期间出现备战。', {'kind': 'battle_popup'})
        current, metadata = self.snapshot(fresh)
        fit = self.request(current, previous=name)
        if not fit.get('accepted'):
            raise ValueError('原型输入前无法复核双帧几何，旧计划已失效。')
        matrix = np.asarray(fit['matrix'], float)
        hull = np.asarray(fit.get('support', []), np.float32)
        if len(hull) < 3 or any(cv2.pointPolygonTest(hull, tuple(map(float, value)), False) < 0
                                for value in (point, ground)):
            raise ValueError('原型落点或小队不在本次几何支持域内，未外推输入。')
        points = np.array([point, ground], np.float32)[None]
        projected = cv2.perspectiveTransform(points, matrix)[0]
        if (np.linalg.norm(projected[0] - point) > 8.
                or np.linalg.norm(projected[1] - metadata['ground']) > 12.):
            raise ValueError('原型推理期间镜头或小队发生变化，未发送旧落点。')
        return fresh, current, metadata

    def settle(self, previous, before, target):
        """通过背景对应扣除镜头位移，再确认小队真实移动；停稳本身不等于传送成功。"""
        session = self.runner.session
        deadline = time.monotonic() + 8.
        movements = []
        while time.monotonic() < deadline:
            session.pause(.5)
            field = self.runner.map_frame()
            if goto.battle_popup_score(field) > .8:
                return 'entry'
            current, metadata = self.snapshot(field)
            fit = self.request(current, previous=previous)
            if not fit.get('accepted'):
                session.record_evidence('learned_feedback_unknown', field, registration=fit, arrays=dict(field=field))
                raise ValueError('原型动作后静态几何未确认，不能推断移动或设施连接。')
            matrix = np.asarray(fit['matrix'], float)
            old = cv2.perspectiveTransform(np.float32([[before['ground']]]), matrix)[0, 0]
            moved = float(np.linalg.norm(np.asarray(metadata['ground']) - old))
            movements.append(moved)
            session.record_evidence('learned_movement_feedback', field, movement_px=round(moved, 2),
                                    target=target, registration=fit, transfer='unconfirmed', arrays=dict(field=field))
            if len(movements) >= 3 and np.ptp(movements[-3:]) < 6.:
                if min(movements[-3:]) < 16.:
                    raise ValueError('原型动作后小队未确认真实移动，不将停稳计为通行成功。')
                logger.info(f'学习型设施原型确认小队移动约{moved:.1f}px；设施连接仍保持未确认。')
                return 'attempted'
        raise ValueError('原型动作八秒内没有稳定移动反馈，停止并保留现场。')

    def run(self, field):
        """有限执行一个镜头或短步提议；无表面支持时停止，不退回盲点设施。"""
        from module.campaign.structures import StructureUnconfirmed
        runner, session = self.runner, self.runner.session
        if len(runner.owner.story_wins) != self.wins:
            self.steps, self.wins = 0, len(runner.owner.story_wins)
        if self.steps >= 8:
            raise StructureUnconfirmed('学习型设施原型八次动作仍未恢复胜利，停止并保留现场。')
        try:
            name, metadata = self.snapshot(field)
            plan = self.request(name)
            session.record_evidence('learned_facility_plan', field, plan=plan, arrays=dict(field=field))
            session.preview(field)
            logger.info(f"学习型设施原型：{plan.get('action')}，{plan.get('reason')}，"
                        f"推理{plan.get('seconds', 0):.2f}秒，峰值显存{plan.get('peak_allocated_mb', 0):.0f}MB。")
            if plan.get('action') == 'needs_review':
                raise ValueError('原型没有确认支持的设施接近路径。')
            if plan.get('action') in ('move', 'interact') and plan.get('surface_validated') is not True:
                raise ValueError('原型表面缺少语义核对，未发送地面输入。')
            point = plan.get('click', metadata['ground'])
            _, current, before = self.recheck(name, field, point, metadata['ground'])
            session.begin_target('learned_facility')
            self.steps += 1
            if plan['action'] == 'pan':
                pan_scene(session, np.asarray(plan['drag'], float), squad_ground=np.asarray(before['ground']))
                runner.memory.previous = None
                return 'scouted'
            if plan['action'] not in ('move', 'interact'):
                raise ValueError('未知原型动作，未发送输入。')
            session.tap(plan['click'])
            result = self.settle(current, before, plan.get('target'))
            runner.memory.previous = None
            runner.deferred.clear()
            # 踩过机关或走过设施后道路可能已通，之前走不到的位置重新允许尝试。
            if hasattr(runner, 'rejected'):
                runner.rejected.clear()
            return 'scouted' if plan['action'] == 'move' and result != 'entry' else result
        except TargetTriggered:
            return 'entry'
        except (OSError, ValueError, RuntimeError) as exc:
            raise StructureUnconfirmed(str(exc)) from exc


def recover_if_enabled(runner, field):
    """原型关闭时返回None；开启时所有拒绝直接保留，不默默回退旧猜测分支。"""
    if not enabled(runner):
        return None
    if not hasattr(runner, '_learned_recovery'):
        runner._learned_recovery = Recovery(runner)
    return runner._learned_recovery.run(field)
