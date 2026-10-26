"""可见设施的有界恢复：两帧确认后尝试开关或平台，不将点击当作开门或传送成功。"""

import time
import json

import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene, scene_unchanged
from module.campaign.navigation.facility_map import (
    gaze_position, in_seek_view, locate_view, record_map_positions, seek_map_point)
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.navigation.hybrid_localization import locate_squad, position_feedback
from module.campaign.facility_memory import FacilityMemory, FacilityUnconfirmed, stable_facilities
from module.campaign.map.catalog import chapter_map
from module.campaign.map.annotations import validate_annotations
from module.campaign.vision.enemy_ring import clickable_ring
from module.campaign.vision.objects import detect_objects, nearby_platforms, visible_platforms
from module.campaign.vision.perception import minimap_squad_ground, squad_arrow
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


def wait_structure_settle(runner, before=None, recenter=True, samples=None):
    """设施落点后共用独立移动反馈；可靠静止才恢复扫描，不据此认领传送成功。

    无有效反馈最多20秒，真实移动可续等但总计60秒；不以整张小地图的低变化比例
    掩盖小范围道路滚动，也不让浮动箭头动画阻止可靠圆心与道路提供静止证据。
    全程看不到小队时用已核验的离屏头像找回镜头一次，找回后重新计时等一次；仍看不到才停止。
    实录 2026-10-22（22-36）：按目录坐标移镜头点远处开关后，小队走出画面、小地图也没有圆环，
    20秒只得到 squad_unobserved 就中止了整轮。
    小队在画面里但贴边、背景比对覆盖不到时，把镜头朝小队平移一次，同样重新计时等一次。
    实录 2026-10-23（23-31）：踩亮左侧开关后小队停在画面左缘（x≈75），背景角点包不住小队、
    小地图圆环也没读到，20秒只得到 squad_outside_background_support 就中止了整轮。
    samples 不为 None 时收集等待途中看得到小队的帧 (帧, 箭头)，最多 TRANSIT_SAMPLES 帧，供判断中途换乘。
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
        arrow = squad_arrow(field)
        if samples is not None and arrow is not None and len(samples) < TRANSIT_SAMPLES:
            samples.append((field, arrow))
        observation = combine_motion(motion.observe(field, arrow), minimap.observe(field))
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
    arrow = None if field is None else squad_arrow(field)
    if (recenter and state == 'observation_unknown' and arrow is not None
            and observation.get('reason') == 'squad_outside_background_support'):
        ground = np.asarray(arrow, float)+[0., goto.ARROW_GROUND_OFFSET]
        drag = np.asarray(goto.VIEWPORT_PIXEL)-ground
        length = float(np.linalg.norm(drag))
        if length >= 120.:
            logger.info('设施交互后小队贴在画面边缘、背景比对覆盖不到：镜头朝小队平移一次后重新等待停稳。')
            pan_scene(runner.session, drag*min(1., 480./length), squad_ground=ground)
            return wait_structure_settle(runner, recenter=False, samples=samples)
    if recenter and state == 'observation_unknown' and observation.get('reason') == 'squad_unobserved':
        logger.info('设施交互后一直看不到小队：找回小队镜头后重新等待停稳一次。')
        recenter_squad(runner)
        if squad_arrow(runner.map_frame()) is not None:
            return wait_structure_settle(runner, recenter=False, samples=samples)
    # 停止策略分级（2026-10-25）：停稳只决定何时重新扫描，确认不了不是风险；下一轮照常从新鲜画面判断，
    # 小队若仍在走，下一次点击只会改变去向。旧做法在这里整轮停止（5 天定时运行 4 次）。
    logger.info('设施交互后未确认小队停稳：按当前画面继续，下一轮重新扫描敌人与道路。')
    return runner.map_frame() if field is None else field


# 每个受阻目标最多尝试几个探索发现的守关敌人；每次尝试后允许为同一目标再探索一次。
GATE_ENEMY_TRIES = 3
# 受阻后的探索时限（秒）：从小队出发由近及远找关闭机关，看完仍没有才找守关敌人。
EXPLORE_SECONDS = 300.
# 同一受阻目标的总时限；超过即停止并报告，不等任务一小时总时限。
BLOCKED_GOAL_SECONDS = 900.
# 设施目录不可用时，每次真实进展（胜利或开关确认开启）前最多交互几次设施。
# 目录可用时每个设施在同一道路状态下只试一次（跨运行保存），由受阻总时限收口。
STRUCTURE_INPUTS = 2


# 平台尝试后小队离目标平台最近距离超过它（客户区像素）、且换了位置，算中途换乘停在半路。
TRANSIT_NEAR = 150.
# 判断中途换乘最多保留的等待帧数（全尺寸帧约 5MB 一张）。
TRANSIT_SAMPLES = 24


# 点关闭机关后盯住机关本身的时长，以及原位判定半径（客户区像素）。
SWITCH_WATCH_SECONDS = 20.
SWITCH_NEAR = 40.


def recenter_squad(runner):
    """小队不在画面时点击已核验的离屏头像找回镜头；地图恢复失败按设施未确认停止。"""
    from module.campaign.vision.minimap_occlusion import recenter_offscreen
    # 实录 2026-10-22（22-27）：任务定位后小队就站在光幕旁、仍在画面内，没有离屏头像；
    # 原先一律报“未读到稳定的离屏小队头像”，看起来像找回失败。
    if squad_arrow(runner.map_frame()) is not None:
        logger.info('小队已在画面内，从当前镜头开始。')
        runner.memory.previous = None
        return None
    try:
        marker = recenter_offscreen(runner.session)
    except RuntimeError as exc:
        raise StructureUnconfirmed(f'找回小队镜头失败：{exc}')
    if marker is None:
        logger.info('未读到稳定的离屏小队头像，镜头保持原位。')
    runner.memory.previous = None
    return marker


def switch_look(runner, catalog, blocked_goal, step, goal=False):
    """只找关闭机关地看当前一屏；返回 (entry/found/none, 机关, 新鲜帧)。"""
    from module.campaign.explorer import Explorer
    explorer = Explorer(runner, want=('switch',), catalog=catalog, skip=runner.switch_skip, goal=blocked_goal)
    seen = explorer.look(step, goal=goal)
    if seen['state'] == 'entry':
        return 'entry', None, seen['field']
    if seen.get('switches'):
        return 'found', seen['switches'][0], seen['field']
    return 'none', None, seen['field']


def goal_view_switch(runner, catalog, blocked_goal):
    """第零级：先只在当前这一屏找关闭机关，再转探索。

    实录 2026-10-21（22-21）：开关就在任务菱形右侧、当前画面里已检测到（0.95），但受阻后
    螺旋以小队为中心，两三屏就停在光幕后的敌人和远处另一个开关上，一直没看这一屏。
    同一道路状态下每个受阻目标只做一次；返回None（已看过）或dict(state=entry/found/none, ...)。
    """
    if not runner.world.once('goal_view', blocked_goal, reset_on=('switch',)):
        return None
    state, target, fresh = switch_look(runner, catalog, blocked_goal, 'goal', goal=True)
    if state == 'entry':
        return dict(state='entry', field=fresh)
    runner.session.record_evidence('goal_view_result', fresh, state=state, target=target, arrays=dict(field=fresh))
    if state == 'found':
        logger.info(f"任务目标旁发现关闭机关 {target['position']}（{target['confidence']:.2f}），交由原生寻路前往。")
        return dict(state='found', target=target, field=fresh)
    logger.info('任务目标旁没有可用的关闭机关，回到小队附近继续探索。')
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
    if field is not None:
        runner.session.record_evidence('switch_watch_end', field, target=target, missing=missing,
                                       squad=squad_arrow(field) is not None, arrays=dict(field=field))
    return field, 'unconfirmed'


def reset_switch_marks(runner, catalog):
    """确认踩亮一个开关：道路版本号加一，之前的受阻、已试、已看记录随之失效。

    用户要求（2026-10-22）：在碰到下一个开关前可以留标记，一旦触碰开关就全部重置，因为之前
    走不到的地方都可能已经能到了。开关在原位变为开启是已验证的道路状态改变，不是参数放宽。
    """
    runner.world.advance('switch')
    if catalog is not None:
        catalog.switch_activated()
    logger.info('开关已确认开启：道路版本号加一，受阻标记（打不到的敌人、已看屏、已试设施）全部失效。')


# 每个受阻目标最多去看几个目录里的已知关闭机关（每个约几次配准加拖动，二三十秒）。
KNOWN_SWITCH_TRIES = 3


def known_switches(catalog, blocked_goal, tried):
    """目录里有底图坐标、未确认开启、对本受阻目标未试过的地面机关编号与坐标。"""
    return [(identifier, np.asarray(record['map_position'], float))
            for identifier, record in catalog.data['facilities'].items()
            if record['kind'] == 'ground_mechanism' and 'map_position' in record
            and record.get('activated') != 'observed_on' and identifier not in tried
            and catalog.eligible(identifier, blocked_goal)]


def known_switch_view(runner, catalog, blocked_goal):
    """第负一级：按底图坐标把镜头移到目录里已知、未用过的关闭机关处，当场两帧确认后交回。

    用户反馈（2026-10-22，22-36）：开关离起点很远，点感叹号敌人会被引导到错误方向，螺旋只看
    小队附近；目录里既然知道这个机关没用过，就应该直接去试。坐标只决定往哪看，点击仍用
    新鲜检测；按镜头注视点由近及远，每个受阻目标最多看 KNOWN_SWITCH_TRIES 个。
    返回 None（没有可看的或都没确认到，镜头已找回小队）或 dict(state=entry/found, ...)。
    """
    if catalog is None:
        return None
    world = runner.world
    tried = world.members('known_switch_tried', blocked_goal, reset_on=('switch',))
    if len(tried) >= KNOWN_SWITCH_TRIES or not known_switches(catalog, blocked_goal, tried):
        return None
    try:
        gaze = gaze_position(locate_view(runner, 'known_switch_start'))
    except ValueError as exc:
        logger.info(f'目录里有未用过的已知机关，但镜头配准未通过：{exc}')
        return None
    moved = False
    while len(tried) < KNOWN_SWITCH_TRIES:
        pool = known_switches(catalog, blocked_goal, tried)
        if not pool:
            break
        identifier, point = min(pool, key=lambda item: np.linalg.norm(item[1]-gaze))
        tried = world.add_member('known_switch_tried', blocked_goal, identifier, reset_on=('switch',))
        logger.info(f'目标受阻：目录里的关闭机关 {identifier} 未对本目标试过，按底图坐标 '
                    f'{np.round(point).astype(int).tolist()} 移镜头过去用新鲜检测确认。')
        moved = True
        reached = seek_map_point(runner, point, tag='known_switch')
        if reached is None:
            continue
        state, target, fresh = switch_look(runner, catalog, blocked_goal, ('catalog', identifier))
        runner.session.record_evidence('known_switch_view', fresh, facility_id=identifier,
                                       map_position=point.tolist(), expected=np.asarray(reached['client']).tolist(),
                                       state=state, target=target, arrays=dict(field=fresh))
        if state == 'entry':
            return dict(state='entry', field=fresh)
        if state == 'found':
            logger.info(f"已知机关附近新鲜确认关闭机关 {target['position']}（{target['confidence']:.2f}，"
                        f"预计 {np.round(reached['client']).astype(int).tolist()}），交由原生寻路前往。")
            return dict(state='found', target=target, field=fresh)
        logger.info(f'已知机关 {identifier} 附近未新鲜确认到可用的关闭机关。')
    if moved:
        recenter_squad(runner)
    return None


# 每个受阻目标最多为目录里没试过的电梯平台移几次镜头（每次约十秒）。
KNOWN_PLATFORM_LOOKS = 3


def known_platforms(catalog, blocked_goal, tried):
    """目录里有底图坐标、对本受阻目标未试过的电梯平台编号与坐标。"""
    return [(identifier, np.asarray(record['map_position'], float))
            for identifier, record in catalog.data['facilities'].items()
            if record['kind'] == 'elevator' and record.get('map_position') is not None
            and identifier not in tried and catalog.eligible(identifier, blocked_goal)]


def eligible_structures(catalog, candidates, registered, blocked_goal):
    """两帧稳定的候选对应到目录编号，只留对本受阻目标没试过的。"""
    return [dict(item, facility_id=known['facility_id']) for item in candidates
            for known in registered if known['kind'] == item['kind']
            and np.linalg.norm(np.asarray(known['position'])-item['position']) < 1.
            and catalog.eligible(known['facility_id'], blocked_goal)]


def platform_look(runner, catalog, blocked_goal, expected):
    """移镜头后两帧确认平台；返回 (entry/found/none, 离预计位置最近的未试平台, 新鲜帧)。"""
    session = runner.session
    field = runner.map_frame()
    first = structure_observations(field, squad_arrow(field), True)
    session.pause(.25)
    fresh = runner.map_frame()
    if goto.battle_popup_score(fresh) > .8:
        return 'entry', None, fresh
    second = structure_observations(fresh, squad_arrow(fresh), True)
    try:
        registered = catalog.observe(fresh, stable_facilities(first, second))
    except FacilityUnconfirmed:
        return 'none', None, fresh
    pool = [item for item in eligible_structures(catalog, stable_structures(first, second), registered, blocked_goal)
            if item['kind'] == 'elevator']
    if not pool:
        return 'none', None, fresh
    return 'found', min(pool, key=lambda item: np.linalg.norm(np.asarray(item['position'])-expected)), fresh


def known_platform_view(runner, catalog, blocked_goal):
    """最后一级：画面内没有可试设施时，按底图坐标去看目录里没试过的电梯平台，当场两帧确认后交回。

    实录 2026-10-25（HARD 9）：点 Boss 时游戏把小队经右侧炮台平台送上山脊，山脊是死路；之后的受阻
    恢复只看当前画面，画面里唯一的平台以前试过，于是直接停止，炮台区另外几个平台（目录里 11 个
    没试过）一次没踩。由近及远，每个受阻目标在同一道路状态下最多移 KNOWN_PLATFORM_LOOKS 次镜头；
    坐标只决定往哪看，点击仍用新鲜检测，传送与否由之后的实际寻路判断。
    返回 None（没有可看的或都没确认到，镜头已找回小队）或 dict(state=entry/found, ...)。
    """
    if catalog is None:
        return None
    world = runner.world

    def looks():
        return world.count('known_platform_look', blocked_goal, reset_on=('switch',))

    tried = world.members('known_platform_tried', blocked_goal, reset_on=('switch',))
    if looks() >= KNOWN_PLATFORM_LOOKS or not known_platforms(catalog, blocked_goal, tried):
        return None
    try:
        gaze = gaze_position(locate_view(runner, 'known_platform_start'))
    except ValueError as exc:
        logger.info(f'目录里有没试过的电梯平台，但镜头配准未通过：{exc}')
        return None
    moved = False
    while looks() < KNOWN_PLATFORM_LOOKS:
        pool = known_platforms(catalog, blocked_goal, tried)
        if not pool:
            break
        identifier, point = min(pool, key=lambda item: np.linalg.norm(item[1]-gaze))
        tried = world.add_member('known_platform_tried', blocked_goal, identifier, reset_on=('switch',))
        world.bump('known_platform_look', blocked_goal, reset_on=('switch',))
        logger.info(f'目标受阻：目录里的电梯平台 {identifier} 未对本目标试过，按底图坐标 '
                    f'{np.round(point).astype(int).tolist()} 移镜头过去用新鲜检测确认。')
        moved = True
        reached = seek_map_point(runner, point, tag='known_platform')
        if reached is None:
            continue
        state, target, fresh = platform_look(runner, catalog, blocked_goal, np.asarray(reached['client'], float))
        runner.session.record_evidence('known_platform_view', fresh, facility_id=identifier,
                                       map_position=point.tolist(), expected=np.asarray(reached['client']).tolist(),
                                       state=state, target=target, arrays=dict(field=fresh))
        if state == 'entry':
            return dict(state='entry', field=fresh)
        if state == 'found':
            logger.info(f"已知平台附近新鲜确认没试过的平台 {target['facility_id']} {target['position']}"
                        f"（{target['confidence']:.2f}），交由原生寻路前往。")
            return dict(state='found', target=target, field=fresh)
        # 实录 2026-10-25（HARD 9）：scene-0004/0007/0008 的底图坐标都投影在小队身旁，画面里只有一个已试平台，
        # 三次查看拍的是同一屏。一屏内没确认到的其余目录平台多是配准误差留下的重复编号，一并排除。
        covered = sorted(other for other, spot in pool
                         if other != identifier and in_seek_view(reached['chart'], reached['report'], spot))
        for other in covered:
            tried = world.add_member('known_platform_tried', blocked_goal, other, reset_on=('switch',))
        logger.info(f'已知平台 {identifier} 附近未新鲜确认到没试过的平台'
                    + (f"；同屏范围内的 {', '.join(covered)} 一并排除" if covered else '') + '。')
    if moved:
        recenter_squad(runner)
    return None


def attack_gate_enemy(runner, searched, blocked_goal):
    """点击探索发现的未清敌人脚圈，按原生寻路反馈等待备战；返回'entry'或'scouted'。

    开战与否只看真实备战弹窗，关卡身份由战斗流程读标题复核；停滞则记下位置不再点同一处。
    实录 2026-10-21（21-32）：首领两侧普通敌人和首领同在光幕后，搜索先选中它们，走到光幕停住后
    整次搜索预算也被用掉，关闭机关一直没被找过。敌人尝试不占探索次数，达到上限后只找机关。
    """
    session, fresh = runner.session, searched['field']
    target = searched['targets'][0] if 'targets' in searched else searched['target']
    click = list(target['click'] if 'click' in target else target['position'])
    runner.world.bump('gate_enemy', blocked_goal, reset_on=('switch',))
    runner.world.put('switch_hunt', blocked_goal, False)
    session.begin_target('gate_enemy')
    session.record_evidence('gate_enemy_attempt', fresh, target=target, arrays=dict(field=fresh))
    logger.info(f'目标受阻，先打探索发现的未清敌人 {click}；开门须等首领目标重新尝试时确认。')
    try:
        session.tap(click)
    except TargetTriggered:
        return 'entry'
    if runner.wait_entry(fresh, arrow=squad_arrow(fresh), target=dict(target, click=click)):
        return 'entry'
    runner.rejected.add(fresh, click, 'stalled')
    runner.memory.previous = None
    return 'scouted'


def closest_approach(catalog, identifier, samples):
    """等待途中小队离该设施的最近距离（客户区像素）；没有一帧能投影设施时返回 None。"""
    closest = None
    for field, arrow in samples:
        position = catalog.project_facility(field, identifier)
        if position is None:
            continue
        ground = np.asarray(arrow, float)+[0., goto.ARROW_GROUND_OFFSET]
        distance = float(np.linalg.norm(np.asarray(position, float)-ground))
        closest = distance if closest is None else min(closest, distance)
        if closest <= TRANSIT_NEAR:
            break
    return closest


def continue_transit(runner, target, before, observed, samples, catalog, blocked_goal, resumed):
    """平台尝试后小队换了位置、却一直没到过目标平台旁：多半中途坐了别的传送平台，游戏到站即停。

    实录 2026-10-25（HARD 9）：从山脊点炮台区左侧平台，小队经山脊传送平台到了楼顶就停下，离目标约
    525px；脚本把左侧平台记成已试，转去点走不到的敌人，又被游戏送回山脊。人工做法是到站后再点一次。
    小队到过目标旁（坐的就是它）或画面没变（没走动）都不算；投影不出目标时判断不了，按已试处理。
    返回 None（不是中途换乘或无法续点）或续点的 attempt_structure 结果。
    """
    if scene_unchanged(before, observed):
        return None
    closest = closest_approach(catalog, target['facility_id'], samples+[(observed, squad_arrow(observed))])
    if closest is None or closest <= TRANSIT_NEAR:
        return None
    catalog.transit(target['facility_id'])
    if resumed:
        logger.info(f"续点后小队仍停在离平台 {target['facility_id']} {closest:.0f}px 处：不再续点，按已试处理。")
        return None
    field = runner.map_frame()
    first = structure_observations(field, squad_arrow(field), True)
    runner.session.pause(.25)
    fresh = runner.map_frame()
    if goto.battle_popup_score(fresh) > .8:
        return 'entry'
    second = structure_observations(fresh, squad_arrow(fresh), True)
    try:
        registered = catalog.observe(fresh, stable_facilities(first, second))
    except FacilityUnconfirmed:
        registered = []
    # 续点是同一次尝试的延续，只核对目标平台仍在画面里，不再按已试记录过滤。
    again = [dict(item, facility_id=known['facility_id']) for item in stable_structures(first, second)
             for known in registered if known['facility_id'] == target['facility_id']
             and known['kind'] == item['kind'] and np.linalg.norm(np.asarray(known['position'])-item['position']) < 1.]
    runner.session.record_evidence('structure_transit', fresh, target=target, closest=closest,
                                   resumed=bool(again), arrays=dict(field=fresh))
    logger.info(f"小队换了位置但一直没到平台 {target['facility_id']} 旁（最近 {closest:.0f}px）：多半中途坐了别的"
                '传送平台、到站即停。' + ('从新位置再点一次。' if again else '画面里没新鲜确认到它，受阻时可再试一次。'))
    if not again:
        return None
    return attempt_structure(runner, again[0], fresh, catalog, blocked_goal, resumed=True)


def attempt_structure(runner, target, fresh, catalog, blocked_goal, located=None, resumed=False):
    """踩关闭机关或走平台：发送一次新鲜落点，等原位开启或停稳；返回'entry'或'attempted'。

    located 是交互前的地图定位（只有每次进展前第一次设施尝试才补），交互后再定位一次比较。
    机关原位确认开启算道路改变（switch），其余交互结果保持未确认、只算走过设施（facility）。
    平台尝试停在半路（中途换乘）时从新位置续点一次，resumed 表示这已经是续点。
    """
    session = runner.session
    session.begin_target('structure_attempt')
    session.record_evidence('structure_attempt', fresh, target=target, arrays=dict(field=fresh))
    runner.world.bump('structure_input', reset_on=('win', 'switch'))
    if 'search_key' in target:
        runner.switch_skip.add(target['search_key'])
    if catalog is not None and 'facility_id' in target:
        catalog.attempt(target['facility_id'], blocked_goal)
    logger.info(f"尝试设施：{target['kind']}／{target['state']}，位置 {target['position']}；"
                '交互结果保持未确认，下一步重新扫描实际道路和敌人。')
    try:
        session.tap(target['position'])
    except TargetTriggered:
        return 'entry'
    switch_state = 'unconfirmed'
    samples = [] if target['kind'] == 'elevator' and catalog is not None and 'facility_id' in target else None
    if target['kind'] == 'ground_mechanism':
        observed, switch_state = watch_switch(runner, target)
        if switch_state == 'entry':
            return 'entry'
        if observed is not None and squad_arrow(observed) is None:
            recenter_squad(runner)
        observed = wait_structure_settle(runner)
    else:
        observed = wait_structure_settle(runner, fresh, samples=samples)
    location_result = dict(state='unknown', transfer='unconfirmed', activated=(
        'confirmed' if switch_state == 'activated' else 'unconfirmed'))
    if located is not None and goto.battle_popup_score(observed) <= .8:
        after_location = locate_squad(session, 'structure_after_location')
        location_result = position_feedback(located, after_location)
        observed = runner.map_frame()
        session.record_evidence('structure_position_result', observed, before=located,
                                after=after_location, result=location_result, arrays=dict(field=observed))
    session.record_evidence('structure_observed', observed, attempted=target,
                            state='unconfirmed', position_feedback=location_result, arrays=dict(field=observed))
    if catalog is not None and 'facility_id' in target:
        catalog.feedback(target['facility_id'], location_result)
    # 镜头或小队可能已经移动，旧观察身份不得阻止新画面的独立目标复核。
    runner.memory.previous = None
    if switch_state == 'activated':
        reset_switch_marks(runner, catalog)
        return 'attempted'
    # 踩过机关或走过设施后道路可能已通：之前走不到的位置、暂缓目标都重新允许尝试。
    runner.world.advance('facility')
    if samples is not None and goto.battle_popup_score(observed) <= .8:
        resumed = continue_transit(runner, target, fresh, observed, samples, catalog, blocked_goal, resumed)
        if resumed is not None:
            return resumed
    return 'attempted'


def recover_visible_structure(runner, field, allow_unanchored=False):
    """普通目标受阻后：先看画面内设施，再按「目录已知机关 → 目标旁 → 探索」找关闭机关，
    画面内仍没有可试设施时去看目录里没试过的电梯平台。

    平台 start/end 只是模型外观类别，不代表可逆或已经传送。游戏自身决定路径和交互；
    不编造电梯配对、地面坐标或成功结果。目录可用时每个设施在同一道路状态下只试一次，
    目录不可用时每次真实进展前最多交互 STRUCTURE_INPUTS 次；都试过或超过受阻总时限才停止。
    小队位置读不到时用画面中心排序（停止策略分级 2026-10-25）。
    """
    learned = recover_if_enabled(runner, field)
    if learned is not None:
        return learned
    session, world = runner.session, runner.world
    catalog = facility_catalog(runner)
    blocked_goal = 'quest:'+str(runner.quest)
    since = world.get('blocked_since', blocked_goal, reset_on=('switch',))
    if since is None:
        since = world.put('blocked_since', blocked_goal, time.monotonic())
    elapsed = time.monotonic()-since
    if elapsed > BLOCKED_GOAL_SECONDS:
        raise StructureUnconfirmed(f'目标受阻已 {elapsed/60:.0f} 分钟，目录机关、探索和设施尝试均未打通，'
                                   '停止并保留现场与设施目录。')
    first = structure_observations(field, squad_arrow(field), allow_unanchored)
    session.pause(.25)
    fresh = runner.map_frame()
    if goto.battle_popup_score(fresh) > .8:
        return 'entry'
    arrow = squad_arrow(fresh)
    second = structure_observations(fresh, arrow, allow_unanchored or arrow is None)
    candidates = stable_structures(first, second)
    inputs = world.count('structure_input', reset_on=('win', 'switch'))
    session.record_evidence('structure_probe', fresh, objects=second, candidates=candidates,
                            previous_objects=first, arrays=dict(field=fresh, previous_field=field), inputs=inputs)
    session.preview(fresh)
    registered = []
    if catalog is not None:
        try:
            registered = catalog.observe(fresh, stable_facilities(first, second))
        except FacilityUnconfirmed as exc:
            # 目录只提供去重和历史；无法重定位时不新建编号，本轮由单目标预算限制重复尝试。
            session.record_evidence('facility_catalog_unrelocated', fresh, reason=str(exc))
            catalog = None
        moved = record_map_positions(runner, catalog, registered, fresh)
        if moved is not None and not moved['unchanged']:
            # 开合地图后画面变了：本帧候选的像素不再可用，下一轮重新扫描。
            runner.memory.previous = None
            return 'scouted'
    if catalog is not None:
        candidates = eligible_structures(catalog, candidates, registered, blocked_goal)
    # 目录按设施去重：同一道路状态下每个设施只试一次，次数不再另设上限。
    budget = catalog is not None or inputs < STRUCTURE_INPUTS
    ground = None if arrow is None else np.asarray(arrow) + [0., goto.ARROW_GROUND_OFFSET]
    if ground is None:
        ground = minimap_squad_ground(fresh)
        if ground is not None:
            logger.info(f'小队头顶箭头未读到，按小地图圆环推算小队在 {np.round(ground).astype(int).tolist()}。')
    anchor = np.asarray(goto.VIEWPORT_PIXEL, float) if ground is None else ground
    # 邻近平台仍可能需要主动踏上或交互，不能以100px近距推定已经操作；距离只用于排序。
    candidates = [item for item in candidates if np.linalg.norm(np.asarray(item['position'])-anchor) <= 1000.]
    searched = None
    if (budget and not any(item['kind'] == 'ground_mechanism' for item in candidates)
            and world.once('switch_hunt', blocked_goal, reset_on=('switch',))):
        # 门旁平台可能只是单向到达端；先找关闭机关，找不到再退回平台尝试。
        searched = known_switch_view(runner, catalog, blocked_goal)
        if searched is None:
            searched = goal_view_switch(runner, catalog, blocked_goal)
        if searched is not None and searched['state'] == 'entry':
            return 'entry'
        if searched is None or searched['state'] != 'found':
            if searched is not None:
                recenter_squad(runner)
            from module.campaign.explorer import Explorer
            enemies = world.count('gate_enemy', blocked_goal, reset_on=('switch',)) < GATE_ENEMY_TRIES
            searched = Explorer(runner, want=('switch', 'enemy') if enemies else ('switch',), catalog=catalog,
                                seconds=EXPLORE_SECONDS, skip=runner.switch_skip, goal=blocked_goal).run()
            session.record_evidence('switch_search_result', searched.get('field', fresh),
                                    **{key: value for key, value in searched.items()
                                       if key not in ('field', 'targets')})
        if searched['state'] == 'entry':
            return 'entry'
        if searched['state'] == 'enemy':
            return attack_gate_enemy(runner, searched, blocked_goal)
        if searched['state'] in ('found', 'switch'):
            fresh, candidates = searched['field'], [searched['target']]
        else:
            # 镜头已离开原位置，旧帧中的平台坐标不再可用；回到下一轮重新扫描与受阻判定。
            recenter_squad(runner)
            runner.memory.previous = None
            return 'scouted'
    if not candidates and budget and searched is None:
        searched = known_platform_view(runner, catalog, blocked_goal)
        if searched is not None and searched['state'] == 'entry':
            return 'entry'
        if searched is not None:
            fresh, candidates = searched['field'], [searched['target']]
    if candidates and budget:
        target = min(candidates, key=lambda item: (
            item['kind'] != 'ground_mechanism', np.linalg.norm(np.asarray(item['position'])-anchor)))
        # 仅每次真实进展前的首个、画面内直接看到的设施尝试补地图证据，探索找到的不再全图定位。
        located = None
        if searched is None and inputs == 0:
            located = locate_squad(session, 'structure_before_location')
            session.pause(.25)
            fresh = runner.map_frame()
            if goto.battle_popup_score(fresh) > .8:
                return 'entry'
            current = structure_observations(fresh, squad_arrow(fresh), True)
            matches = stable_structures([target], current)
            if len(matches) != 1:
                logger.info('地图定位后设施未在新鲜画面保持唯一确认，未发送旧落点；下一轮重新观察。')
                return 'scouted'
            target = dict(matches[0], **({'facility_id': target['facility_id']}
                                        if 'facility_id' in target else {}))
        return attempt_structure(runner, target, fresh, catalog, blocked_goal, located)
    raise StructureUnconfirmed('目标受阻，画面内和目录里的设施都已试过（或交互预算已用尽），探索也没有找到'
                               '可用的关闭机关或敌人；尚未确认可通行路线，停止并保留现场。')
