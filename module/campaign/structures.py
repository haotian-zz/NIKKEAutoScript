"""可见设施的有界恢复：两帧确认后尝试开关或平台，不将点击当作开门或传送成功。"""

import time
import json

import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.navigation.hybrid_localization import locate_squad, position_feedback
from module.campaign.facility_memory import FacilityMemory, FacilityUnconfirmed, stable_facilities
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


# 每个受阻目标最多尝试几个搜索发现的守关敌人；每次开战后允许为同一目标再搜一次。
GATE_ENEMY_TRIES = 3
# 受阻后逐级放大的耗时：任务定位看目标旁（约5秒）→ 回小队螺旋（约1分钟）→ 全章设施巡检
# （一次性，结果持久化）→ 平台与换视角。
SPIRAL_PANS = 8
SPIRAL_SECONDS = 60.
SURVEY_SECONDS = 360.
# 同一受阻目标的总时限；超过即停止并报告，不等任务一小时总时限。
BLOCKED_GOAL_SECONDS = 900.


# 点关闭机关后盯住机关本身的时长，以及原位判定半径（客户区像素）。
SWITCH_WATCH_SECONDS = 20.
SWITCH_NEAR = 40.


def recenter_squad(runner):
    """小队不在画面时点击已核验的离屏头像找回镜头；地图恢复失败按设施未确认停止。"""
    from module.campaign.vision.minimap_occlusion import recenter_offscreen
    try:
        marker = recenter_offscreen(runner.session)
    except RuntimeError as exc:
        raise StructureUnconfirmed(f'找回小队镜头失败：{exc}')
    if marker is None:
        logger.info('未读到稳定的离屏小队头像，镜头保持原位。')
    runner.memory.previous = None
    return marker


def goal_view_switch(runner, catalog, blocked_goal):
    """第零级：点任务定位让游戏把镜头移到受阻目标旁，只在这一屏找关闭机关。

    实录 2026-10-21（22-21）：开关就在任务菱形右侧，任务定位那一帧已检测到（0.95），但受阻后
    螺旋以小队为中心，两三屏就停在光幕后的敌人和远处另一个开关上，一直没看目标旁边。
    每个受阻目标只做一次；返回None（未定位）或dict(state=entry/found/none, ...)。
    """
    from module.campaign.switch_search import SwitchSearch
    views = getattr(runner, 'goal_views', None)
    focus = getattr(runner, 'focus_quest', None)
    if not isinstance(views, set) or blocked_goal in views or focus is None:
        return None
    views.add(blocked_goal)
    focused = focus(runner.map_frame(), 'goal_view')
    if focused == 'entry':
        return dict(state='entry')
    if not focused:
        return None
    search = SwitchSearch(runner, catalog, pans=0, skip=runner.switch_skip, enemies=False)
    state, target, fresh = search.look('goal', goal=True)
    if state == 'entry':
        return dict(state='entry', field=fresh)
    runner.session.record_evidence('goal_view_result', fresh, state=state, target=target, arrays=dict(field=fresh))
    if state == 'found':
        logger.info(f"任务目标旁发现关闭机关 {target['position']}（{target['confidence']:.2f}），交由原生寻路前往。")
        return dict(state='found', target=target, field=fresh)
    logger.info('任务目标旁没有可用的关闭机关，回到小队附近继续螺旋搜索。')
    return dict(state='none', field=fresh)


def watch_switch(runner, target):
    """点关闭机关后先盯机关本身：原位连续两帧读到开启即确认；返回(帧, entry/activated/unconfirmed)。

    实录 2026-10-21（22-21）：开关在任务目标旁、镜头停在那里时小队在画面外，原有停稳判据只得到
    squad_unobserved 而中止。原位连续三帧读不到机关（游戏切镜头或跟随小队）即提前结束观察。
    """
    position = np.asarray(target['position'], float)
    started, on, missing, field = time.monotonic(), 0, 0, None
    while time.monotonic()-started < SWITCH_WATCH_SECONDS:
        runner.session.pause(1.)
        field = runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            return field, 'entry'
        near = [item for item in detect_objects(field, ('ground_mechanism',))
                if item['kind'] == 'ground_mechanism'
                and np.linalg.norm(np.asarray(item['position'])-position) <= SWITCH_NEAR]
        missing = 0 if near else missing+1
        on = on+1 if any(item['state'] == 'on' for item in near) else 0
        if on >= 2:
            runner.session.record_evidence('switch_activated', field, target=target, arrays=dict(field=field))
            logger.info(f"机关已在原位变为开启：{target['position']}；通行仍由后续实际寻路确认。")
            return field, 'activated'
        if missing >= 3:
            break
    return field, 'unconfirmed'


def survey_for_switch(runner, catalog):
    """第二级：全章设施巡检，遇到本受阻目标未试过的关闭机关即停下；返回巡检结果或None。

    实录 2026-10-21（21-32）：开关在光幕东侧烟雾高台，螺旋九次拖动只看到坦克舱盖误检；
    全章巡检排在平台尝试之后，一次都没轮到。巡检不完整只记录原因，已登记部分保留在目录。
    """
    from module.campaign.facility_survey import FacilitySurvey
    from module.campaign.sweep import SweepUnconfirmed
    runner.facility_survey_attempted = True
    try:
        return FacilitySurvey(runner, catalog, seconds=SURVEY_SECONDS, first_off=True).run()
    except (SweepUnconfirmed, FacilityUnconfirmed) as exc:
        logger.info(f'全章设施巡检未完整：{exc}；已登记的设施保留在目录。')
        return None


def attack_gate_enemy(runner, searched, blocked_goal):
    """点击螺旋搜索发现的未清敌人脚圈，按原生寻路反馈等待备战；返回'entry'或'scouted'。

    开战与否只看真实备战弹窗，关卡身份由战斗流程读标题复核；停滞则记下搜索键不再点同一处。
    """
    session, target, fresh = runner.session, searched['target'], searched['field']
    runner.switch_skip.add(target['search_key'])
    tries = runner.gate_enemy_tries
    tries[blocked_goal] = tries.get(blocked_goal, 0) + 1
    # 实录 2026-10-21（21-32）：首领两侧普通敌人和首领同在光幕后，搜索先选中它们，
    # 走到光幕停住后整次搜索预算也被用掉，关闭机关一直没被找过。敌人尝试不占搜索次数，
    # 达到上限后的搜索只找机关（见 recover_visible_structure）。
    runner.switch_searches[blocked_goal] -= 1
    session.begin_target('gate_enemy')
    session.record_evidence('gate_enemy_attempt', fresh, target=target, arrays=dict(field=fresh))
    logger.info(f"目标受阻，先打搜索发现的未清敌人 {target['position']}；开门须等首领目标重新尝试时确认。")
    try:
        session.tap(target['position'])
    except TargetTriggered:
        return 'entry'
    if runner.wait_entry(fresh, arrow=squad_arrow(fresh), target=dict(click=target['position'])):
        return 'entry'
    runner.memory.previous = None
    return 'scouted'


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
    blocked_since = getattr(runner, 'blocked_since', None)
    if isinstance(blocked_since, dict):
        elapsed = time.monotonic()-blocked_since.setdefault(blocked_goal, time.monotonic())
        if elapsed > BLOCKED_GOAL_SECONDS:
            raise StructureUnconfirmed(f'目标受阻已 {elapsed/60:.0f} 分钟，螺旋搜索、全章巡检和设施尝试均未打通，'
                                       '停止并保留现场与设施目录。')
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
        try:
            registered = catalog.observe(fresh, stable_facilities(first, second))
        except FacilityUnconfirmed as exc:
            # 目录只提供去重和历史；无法重定位时不新建编号，本轮由单目标预算限制重复尝试。
            session.record_evidence('facility_catalog_unrelocated', fresh, reason=str(exc))
            registered, catalog = [], None
    if catalog is not None:
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
    searched = None
    searches = getattr(runner, 'switch_searches', None)
    if (isinstance(searches, dict) and ground is not None and runner.structure_inputs < 2
            and not any(item['kind'] == 'ground_mechanism' for item in candidates)
            and searches.get(blocked_goal, 0) < 1):
        # 门旁平台可能只是单向到达端；每个受阻目标先看目标旁，再螺旋找一次关闭机关，找不到再退回平台尝试。
        from module.campaign.switch_search import SwitchSearch
        searched = goal_view_switch(runner, catalog, blocked_goal)
        if searched is not None and searched['state'] == 'entry':
            return 'entry'
        if searched is None or searched['state'] != 'found':
            if searched is not None:
                recenter_squad(runner)
            searches[blocked_goal] = searches.get(blocked_goal, 0) + 1
            enemies = runner.gate_enemy_tries.get(blocked_goal, 0) < GATE_ENEMY_TRIES
            blocked = getattr(runner, 'blocked_click', None)
            searched = SwitchSearch(runner, catalog, pans=SPIRAL_PANS, seconds=SPIRAL_SECONDS,
                                    skip=runner.switch_skip, enemies=enemies,
                                    blocked=[blocked] if isinstance(blocked, (list, tuple, np.ndarray)) else []).run()
            session.record_evidence('switch_search_result', searched.get('field', fresh),
                                    **{key: value for key, value in searched.items() if key != 'field'})
        if searched['state'] == 'entry':
            return 'entry'
        if searched['state'] == 'enemy':
            return attack_gate_enemy(runner, searched, blocked_goal)
        if searched['state'] != 'found' and catalog is not None and not runner.facility_survey_attempted:
            surveyed = survey_for_switch(runner, catalog) or {}
            if surveyed.get('state') == 'entry':
                return 'entry'
            if surveyed.get('state') == 'facility_visible' and 'target' in surveyed:
                fresh = runner.map_frame()
                if goto.battle_popup_score(fresh) > .8:
                    return 'entry'
                logger.info(f"全章巡检发现未试过的关闭机关 {surveyed['target']['position']}，交由原生寻路前往。")
                searched = dict(state='found', field=fresh,
                                target=dict(surveyed['target'], search_key=('survey', surveyed['facility_id'])))
        if searched['state'] == 'found':
            fresh = searched['field']
            candidates = [searched['target']]
        else:
            # 镜头已离开原位置，旧帧中的平台坐标不再可用；回到下一轮重新扫描与受阻判定。
            runner.memory.previous = None
            return 'scouted'
    if candidates and runner.structure_inputs < 2:
        target = (candidates[0] if ground is None else min(candidates, key=lambda item: (
            item['kind'] != 'ground_mechanism', np.linalg.norm(np.asarray(item['position']) - ground))))
        # 仅每次真实胜利前的首个设施尝试补地图证据，普通点击与后续恢复不承担反复全图定位成本。
        before_location = None
        if searched is not None:
            runner.switch_skip.add(target['search_key'])
        elif runner.structure_inputs == 0:
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
        if catalog is not None and 'facility_id' in target:
            catalog.attempt(target['facility_id'], blocked_goal)
        logger.info(f"尝试可见设施：{target['kind']}／{target['state']}，位置 {target['position']}；"
                    '交互结果保持未确认，下一步重新扫描实际道路和敌人。')
        try:
            session.tap(target['position'])
        except TargetTriggered:
            return 'entry'
        switch_state = 'unconfirmed'
        if target['kind'] == 'ground_mechanism':
            observed, switch_state = watch_switch(runner, target)
            if switch_state == 'entry':
                return 'entry'
            if observed is not None and squad_arrow(observed) is None:
                recenter_squad(runner)
            observed = wait_structure_settle(runner)
        else:
            observed = wait_structure_settle(runner, fresh)
        location_result = dict(state='unknown', transfer='unconfirmed', activated=(
            'confirmed' if switch_state == 'activated' else 'unconfirmed'))
        if before_location is not None and goto.battle_popup_score(observed) <= .8:
            after_location = locate_squad(session, 'structure_after_location')
            location_result = position_feedback(before_location, after_location)
            observed = runner.map_frame()
            session.record_evidence('structure_position_result', observed, before=before_location,
                                    after=after_location, result=location_result, arrays=dict(field=observed))
        session.record_evidence('structure_observed', observed, attempted=target,
                                state='unconfirmed', position_feedback=location_result, arrays=dict(field=observed))
        if catalog is not None and 'facility_id' in target:
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
        result = FacilitySurvey(runner, catalog).run() or {}
        if result.get('state') == 'entry':
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
