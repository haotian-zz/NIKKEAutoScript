"""可见设施的有界恢复：两帧确认后尝试开关或平台，不将点击当作开门或传送成功。"""

import time

import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.vision.enemy_ring import clickable_ring
from module.campaign.vision.objects import detect_objects, nearby_platforms
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.native_motion import ClickProgress, MinimapStability, NativeMotion, combine_motion
from module.campaign.vision.learned_recovery import recover_if_enabled
from module.logger import logger


class StructureUnconfirmed(RuntimeError):
    """目标受阻且有界设施探索耗尽，保留未完成并交出现场，禁止继续空跑。"""
    reason = 'structure_interaction_unconfirmed'


def actionable(objects):
    """筛选完整入屏的未开机关及平台；开启机关、屏障、截断框和 HUD 均不作为落点。"""
    return [item for item in objects if clickable_ring(item['box'])
            and ((item['kind'] == 'ground_mechanism' and item['state'] == 'off'
                  and item['confidence'] >= .8)
                 or (item['kind'] == 'elevator' and item['state'] in ('start', 'end')
                     and item['confidence'] >= .9))]


def stable_structures(first, second):
    """两帧同类型、同状态、位置差最多12px且无近邻竞争的设施才成为本次交互候选。"""
    result = []
    for item in actionable(second):
        nearby = [old for old in actionable(first)
                  if old['kind'] == item['kind'] and old['state'] == item['state']
                  and np.linalg.norm(np.asarray(old['position']) - item['position']) <= 12.]
        competition = [other for other in actionable(second) if other is not item
                       and np.linalg.norm(np.asarray(other['position']) - item['position']) < 30.]
        if len(nearby) == 1 and not competition:
            result.append(item)
    return result


def structure_observations(field, arrow):
    """合并网格与可信小队附近的分级补查，平台框IoU≥0.5时保留分数更高的当前观测。

    起终点只是外观分类，重叠去重按平台语义而非标签方向，避免同一平台重复候选互相竞争。
    小队未确认时不建立局部裁图；实际交互仍须两帧原状态与原分数门槛。
    """
    objects = detect_objects(field, ('ground_mechanism', 'elevator', 'barrier'))
    if arrow is None:
        return objects
    local = nearby_platforms(field, np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET])
    for candidate in local:
        duplicates = []
        a = np.asarray(candidate['box'])
        for index, item in enumerate(objects):
            if item['kind'] != 'elevator':
                continue
            b = np.asarray(item['box'])
            intersection = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
            union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
            if intersection/max(1., union) >= .5:
                duplicates.append(index)
        if not duplicates:
            objects.append(candidate)
        elif all(candidate['confidence'] > objects[index]['confidence'] for index in duplicates):
            objects = [item for index, item in enumerate(objects) if index not in duplicates]+[candidate]
    return objects


def wait_structure_settle(runner, before=None):
    """设施落点后共用独立移动反馈；可靠静止才恢复扫描，不据此认领传送成功。

    无有效反馈最多20秒，真实移动可续等但总计60秒；不以整张小地图的低变化比例
    掩盖小范围道路滚动，也不让浮动箭头动画阻止可靠圆心与道路提供静止证据。
    """
    before = runner.map_frame() if before is None else before
    progress = ClickProgress(time.monotonic())
    motion = NativeMotion(before, squad_arrow(before))
    minimap = MinimapStability(before)
    field = None
    while True:
        runner.session.pause(.7)
        field = runner.map_frame(timeout=max(.1, min(30., progress.started + 60. - time.monotonic())))
        if goto.battle_popup_score(field) > .8:
            return field
        observation = combine_motion(motion.observe(field, squad_arrow(field)), minimap.observe(field))
        state = progress.update(time.monotonic(), observation)
        if state == 'stalled':
            runner.session.record_evidence('structure_settled', field, feedback=observation,
                                           state='stationary_interaction_unconfirmed')
            return field
        if state != 'waiting':
            break
    if field is not None:
        runner.session.record_evidence('structure_wait_timeout', field, feedback=observation,
                                       outcome=state, arrays=dict(field=field))
        runner.session.preview(field)
    raise StructureUnconfirmed('设施交互后未确认小队及道路停稳，已保留原图；不发送下一个目标。')


def recover_visible_structure(runner, field):
    """普通目标受阻后尝试可见设施或有界换视角；真实胜利前最多两次交互、四次镜头动作。

    平台 start/end 只是模型外观类别，不代表可逆或已经传送。游戏自身决定路径和交互；
    不编造电梯配对、地面坐标或成功结果。全部输入经正式会话，后续仍须实际备战及胜利确认。
    """
    learned = recover_if_enabled(runner, field)
    if learned is not None:
        return learned
    session = runner.session
    first = structure_observations(field, squad_arrow(field))
    session.pause(.25)
    fresh = runner.map_frame()
    if goto.battle_popup_score(fresh) > .8:
        return 'entry'
    arrow = squad_arrow(fresh)
    second = structure_observations(fresh, arrow)
    candidates = stable_structures(first, second)
    session.record_evidence('structure_probe', fresh, objects=second, candidates=candidates,
                            arrays=dict(field=fresh), inputs=runner.structure_inputs, pans=runner.structure_pans)
    session.preview(fresh)
    if arrow is None:
        raise StructureUnconfirmed('目标受阻，设施恢复未确认画面内小队位置；已保留原图，停止探索。')
    ground = np.asarray(arrow) + [0., goto.ARROW_GROUND_OFFSET]
    # 邻近平台仍可能需要主动踏上或交互，不能以100px近距推定已经操作；距离只用于排序。
    candidates = [item for item in candidates
                  if np.linalg.norm(np.asarray(item['position']) - ground) <= 1000.]
    if candidates and runner.structure_inputs < 2:
        target = min(candidates, key=lambda item: (
            item['kind'] != 'ground_mechanism', np.linalg.norm(np.asarray(item['position']) - ground)))
        session.begin_target('structure_attempt')
        session.record_evidence('structure_attempt', fresh, target=target, arrays=dict(field=fresh))
        runner.structure_inputs += 1
        logger.info(f"尝试可见设施：{target['kind']}／{target['state']}，位置 {target['position']}；"
                    '交互结果保持未确认，下一步重新扫描实际道路和敌人。')
        try:
            session.tap(target['position'])
        except TargetTriggered:
            return 'entry'
        observed = wait_structure_settle(runner, fresh)
        session.record_evidence('structure_observed', observed, attempted=target,
                                state='unconfirmed', arrays=dict(field=observed))
        # 镜头或小队可能已经移动，旧观察身份不得阻止新画面的独立目标复核。
        runner.memory.previous = None
        runner.deferred.clear()
        runner.guided_quests.discard(runner.quest)
        runner.marker_pans = 0
        return 'attempted'
    if runner.structure_pans < 4:
        directions = ((-360., 0.), (0., -240.), (360., 0.), (0., 240.))
        drag = np.asarray(directions[runner.structure_pans])
        runner.structure_pans += 1
        session.begin_target('structure_scout')
        session.record_evidence('structure_scout', fresh, drag=drag.tolist(), arrays=dict(field=fresh))
        pan_scene(session, drag, squad_ground=ground)
        logger.info('目标受阻，已调整一个设施观察视角；镜头变化不记为通关或传送成功。')
        return 'scouted'
    raise StructureUnconfirmed('目标受阻，设施交互及换视角预算已用尽；尚未确认可通行路线，停止并保留现场。')
