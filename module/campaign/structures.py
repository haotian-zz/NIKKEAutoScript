"""可见设施的有界恢复：两帧确认后尝试开关或平台，不将点击当作开门或传送成功。"""

import time
import json

import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.navigation.hybrid_localization import locate_squad, position_feedback
from module.campaign.facility_memory import FacilityMemory, stable_facilities
from module.campaign.map.catalog import chapter_map
from module.campaign.map.annotations import validate_annotations
from module.campaign.vision.enemy_ring import clickable_ring
from module.campaign.vision.objects import detect_objects, nearby_platforms, visible_platforms
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.native_motion import ClickProgress, MinimapStability, NativeMotion, combine_motion
from module.campaign.vision.learned_recovery import recover_if_enabled
from module.logger import logger


class StructureUnconfirmed(RuntimeError):
    """目标受阻且有界设施探索耗尽，保留未完成并交出现场，禁止继续空跑。"""
    reason = 'structure_interaction_unconfirmed'


def facility_catalog(runner):
    """在首次设施恢复时加载实例设施簿；外部标注通过既有底图校验后再导入。

    普通寻敌不承担地图资源与特征内存开销。目录绑定当前章节难度及底图，任务退出时由
    runner生命周期释放内存，磁盘特征仅用于后续重新配准，不授予历史落点输入许可。
    """
    if runner.facilities is None:
        try:
            package, metadata = chapter_map(runner.session.expected_identity[0])
        except OSError:
            package, metadata = None, None
        runner.facilities = FacilityMemory(runner.owner.config.config_name, runner.session.expected_identity,
                                           None if metadata is None else metadata['image_sha256'])
        if package is not None and (package/'annotations.json').exists():
            document = json.loads((package/'annotations.json').read_text(encoding='utf8'))
            validate_annotations(document, metadata['image_sha256'], metadata['size'])
            runner.facilities.import_annotations(document)
    return runner.facilities


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


def structure_observations(field, arrow, unanchored=False):
    """合并网格与可信小队附近的分级补查，平台框IoU≥0.5时保留分数更高的当前观测。

    起终点只是外观分类，重叠去重按平台语义而非标签方向，避免同一平台重复候选互相竞争。
    小队未确认时只有显式unanchored恢复可补查全屏平台；实际交互仍须两帧原状态与原分数门槛。
    """
    objects = detect_objects(field, ('ground_mechanism', 'elevator', 'barrier'))
    if arrow is None and not unanchored:
        return objects
    local = (visible_platforms(field) if arrow is None else
             nearby_platforms(field, np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET]))
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


def recover_visible_structure(runner, field, allow_unanchored=False):
    """普通目标受阻后尝试可见设施或有界换视角；真实胜利前最多两次交互、四次镜头动作。

    平台 start/end 只是模型外观类别，不代表可逆或已经传送。游戏自身决定路径和交互；
    不编造电梯配对、地面坐标或成功结果。全部输入经正式会话，后续仍须实际备战及胜利确认。
    allow_unanchored仅用于稳定任务标记下镜头避让无锚点的恢复：两帧唯一高分平台可交给
    游戏原生寻路；短时最多四帧，仍须相邻两帧通过。没有小队位置时仍禁止拖镜头、按
    距离排序和尝试多个歧义设施，不因观测重试续期进展预算。
    """
    learned = recover_if_enabled(runner, field)
    if learned is not None:
        return learned
    session = runner.session
    catalog = facility_catalog(runner)
    blocked_goal = 'quest:'+str(runner.quest)
    first = structure_observations(field, squad_arrow(field), allow_unanchored)
    session.pause(.25)
    fresh = runner.map_frame()
    if goto.battle_popup_score(fresh) > .8:
        return 'entry'
    arrow = squad_arrow(fresh)
    second = structure_observations(fresh, arrow, allow_unanchored)
    candidates = stable_structures(first, second)
    previous_field, observations = field, 2
    if allow_unanchored and arrow is None and not candidates:
        for _ in range(2):
            previous_field, first = fresh, second
            session.pause(.35)
            fresh = runner.map_frame()
            if goto.battle_popup_score(fresh) > .8:
                return 'entry'
            arrow = squad_arrow(fresh)
            second = structure_observations(fresh, arrow, allow_unanchored)
            observations += 1
            candidates = stable_structures(first, second)
            if candidates:
                break
    session.record_evidence('structure_probe', fresh, objects=second, candidates=candidates,
                            previous_objects=first, observations=observations,
                            arrays=dict(field=fresh, previous_field=previous_field),
                            inputs=runner.structure_inputs, pans=runner.structure_pans)
    session.preview(fresh)
    if catalog is not None:
        registered = catalog.observe(fresh, stable_facilities(first, second))
        candidates = [dict(item, facility_id=known['facility_id']) for item in candidates
                      for known in registered if known['kind'] == item['kind']
                      and np.linalg.norm(np.asarray(known['position'])-item['position']) < 1.
                      and catalog.eligible(known['facility_id'], blocked_goal)]
    if arrow is None and not allow_unanchored:
        raise StructureUnconfirmed('目标受阻，设施恢复未确认画面内小队位置；已保留原图，停止探索。')
    ground = None if arrow is None else np.asarray(arrow) + [0., goto.ARROW_GROUND_OFFSET]
    if ground is None and (len(candidates) != 1 or candidates[0]['kind'] != 'elevator'):
        raise StructureUnconfirmed('缺少小队锚点且未得到两帧唯一高分平台，停止恢复，不发送镜头或走位。')
    # 邻近平台仍可能需要主动踏上或交互，不能以100px近距推定已经操作；距离只用于排序。
    if ground is not None:
        candidates = [item for item in candidates
                      if np.linalg.norm(np.asarray(item['position']) - ground) <= 1000.]
    if candidates and runner.structure_inputs < 2:
        target = (candidates[0] if ground is None else min(candidates, key=lambda item: (
            item['kind'] != 'ground_mechanism', np.linalg.norm(np.asarray(item['position']) - ground))))
        # 仅每次真实胜利前的首个设施尝试补地图证据，普通点击与后续恢复不承担反复全图定位成本。
        before_location = None
        if runner.structure_inputs == 0:
            before_location = locate_squad(session, 'structure_before_location')
            session.pause(.25)
            fresh = runner.map_frame()
            if goto.battle_popup_score(fresh) > .8:
                return 'entry'
            current = structure_observations(fresh, squad_arrow(fresh), allow_unanchored)
            matches = stable_structures([target], current)
            if len(matches) != 1:
                raise StructureUnconfirmed('地图定位后设施未在新鲜画面保持唯一确认，未发送旧落点。')
            target = dict(matches[0], **({'facility_id': target['facility_id']}
                                        if 'facility_id' in target else {}))
        session.begin_target('structure_attempt')
        session.record_evidence('structure_attempt', fresh, target=target, arrays=dict(field=fresh),
                                squad_anchor='unobserved' if ground is None else 'arrow_ground')
        runner.structure_inputs += 1
        if catalog is not None:
            catalog.attempt(target['facility_id'], blocked_goal)
        logger.info(f"尝试可见设施：{target['kind']}／{target['state']}，位置 {target['position']}；"
                    '交互结果保持未确认，下一步重新扫描实际道路和敌人。')
        try:
            session.tap(target['position'])
        except TargetTriggered:
            return 'entry'
        observed = wait_structure_settle(runner, fresh)
        location_result = dict(state='unknown', transfer='unconfirmed', activated='unconfirmed')
        if before_location is not None and goto.battle_popup_score(observed) <= .8:
            after_location = locate_squad(session, 'structure_after_location')
            location_result = position_feedback(before_location, after_location)
            observed = runner.map_frame()
            session.record_evidence('structure_position_result', observed, before=before_location,
                                    after=after_location, result=location_result, arrays=dict(field=observed))
        session.record_evidence('structure_observed', observed, attempted=target,
                                state='unconfirmed', position_feedback=location_result, arrays=dict(field=observed))
        if catalog is not None:
            catalog.feedback(target['facility_id'], location_result)
        # 镜头或小队可能已经移动，旧观察身份不得阻止新画面的独立目标复核。
        runner.memory.previous = None
        runner.deferred.clear()
        runner.guided_quests.discard(runner.quest)
        runner.marker_pans = 0
        return 'attempted'
    if ground is None:
        raise StructureUnconfirmed('无小队锚点的设施交互预算已用尽，停止恢复，不发送镜头动作。')
    if catalog is not None and not runner.facility_survey_attempted and not candidates:
        from module.campaign.facility_survey import FacilitySurvey
        runner.facility_survey_attempted = True
        result = FacilitySurvey(runner, catalog).run()
        if result['state'] == 'entry':
            return 'entry'
        return 'scouted'
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
