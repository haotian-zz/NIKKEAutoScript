"""设施的底图坐标：配准展开小地图得到当前镜头位置，把客户区像素与底图像素互相换算。

关键帧只能在画面有重叠时重定位设施；隔着几屏的已知机关没有办法知道“往哪边拖”。
实录 2026-10-22（22-36）：用户找到的开关离起点很远，螺旋只看小队附近，目录线索也只在
当前画面±300px内才用。底图坐标是全章统一的，记下后可以从任何位置闭环拖镜头过去。

换算按镜头注视点建立：展开 ROI 的 EXPANDED_CENTER 对应客户区 VIEWPORT_PIXEL，局部投影沿用
标定的 A_INV 与 ROI→参考平面单应。结果只用于“去哪里看”，点击仍须当场两帧新鲜检测。
"""

import time

import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import (
    ROI_BOUNDS, camera_drag, inside, pan_scene, scene_unchanged)
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.surface_motion import project
from module.logger import logger

# 闭环拖镜头的次数上限；每次都先重新配准展开地图（约2～3秒）。
SEEK_PANS = 6
# 目标投影进入这块客户区（避开顶部遮挡与底部 HUD）即停止拖动，交给新鲜检测确认。
SEEK_VIEW = (250., 200., 1500., 820.)
# 镜头注视点离目标的底图距离几乎没缩短，视为拖不动（地图边界或拖动被吞）。
SEEK_STUCK = 5.
# 目标落点离已知需手动关卡（EX）底图位置多近就视为同一个敌人：小队触发备战时站在脚圈旁
# 约28底图像素内（controls.SQUAD_REACH），再留一点换算误差。
MANUAL_ENTRY_RADIUS = 30.


def map_localizer(runner):
    """取当前章节的展开地图定位器；主线可见目标会话不预加载底图，首次使用时才加载。"""
    session = runner.session
    if getattr(session, 'localizer', None) is not None:
        return session.localizer
    cached = getattr(runner, 'map_localizer', None)
    if cached is not None:
        return cached
    from module.campaign.session import CampaignLocalizer
    observer, identity = runner.owner.observer, tuple(session.expected_identity)
    if observer.matcher is None or observer.chapter != identity[0]:
        observer.observe(session.win.client(), identity, objects=False)
    if observer.matcher is None:
        return None
    # 只用配准，不规划道路，因此不需要切开图。
    runner.map_localizer = CampaignLocalizer(observer.matcher, None)
    return runner.map_localizer


def gaze_chart(localizer):
    """展开 ROI → 客户区的投影，平移项钉在镜头注视点上；不依赖小队是否在画面里。"""
    affine = np.eye(3)
    affine[:2, :2] = goto.A_INV
    chart = affine @ np.asarray(localizer.old_matrix, float)
    translation = np.eye(3)
    translation[:2, 2] = np.asarray(goto.VIEWPORT_PIXEL, float) - project(chart, goto.EXPANDED_CENTER)
    return translation @ chart


def client_to_map(chart, report, point):
    """客户区像素 → 本次展开地图 ROI → 底图像素。"""
    return project(np.asarray(report['roi_to_map'], float), project(np.linalg.inv(chart), point))


def map_to_client(chart, report, point):
    """底图像素在当前镜头下的客户区位置；不在展开 ROI 里的点返回 None（远处投影不可信）。"""
    roi = project(np.linalg.inv(np.asarray(report['roi_to_map'], float)), point)
    if not inside(roi, ROI_BOUNDS):
        return None
    return project(chart, roi)


def gaze_position(report):
    """镜头注视点的底图坐标。"""
    return project(np.asarray(report['roi_to_map'], float), goto.EXPANDED_CENTER)


def locate_view(runner, tag):
    """展开小地图配准当前镜头视野后收起；配准失败也先收起再抛 ValueError，不影响后续流程。

    小队可以离屏，结果只代表镜头位置。停止信号、窗口变化等会话异常照常上抛，不补发收起。
    """
    localizer = map_localizer(runner)
    if localizer is None:
        raise ValueError('本章底图不可用，无法换算设施坐标。')
    session = runner.session
    session.check()
    goto.map_open(session.win)
    try:
        panel = session.win.capture()
        report = localizer.locate(panel, tag, require_player=False)
        if np.mean(goto.terrain(panel) != goto.terrain(session.win.capture())) > .03:
            raise ValueError('镜头定位期间画面仍在变化。')
    except (RuntimeError, ValueError) as exc:
        goto.map_close(session.win)
        raise ValueError(str(exc)) from exc
    goto.map_close(session.win)
    return report


def record_map_positions(runner, catalog, items, field):
    """本帧刚登记、还没有底图坐标的设施，配准一次展开地图补上坐标；失败只记日志。

    返回 None（无需或未能补坐标）或 dict(field=收起地图后的新帧, unchanged=镜头是否未变)。
    unchanged 为假时调用方不能再用 field 之前那帧的像素点击。
    """
    if catalog is None or not items or not catalog.needs_map(items):
        return None
    try:
        report = locate_view(runner, 'facility_map')
    except ValueError as exc:
        logger.info(f'设施底图坐标未补上：{exc}')
        return None
    chart = gaze_chart(map_localizer(runner))
    errors = catalog.place(items, lambda point: client_to_map(chart, report, point))
    after = runner.map_frame(track=False)
    unchanged = scene_unchanged(field, after)
    runner.session.record_evidence('facility_map_position', after, gaze=gaze_position(report).tolist(),
                                   facilities=[item['facility_id'] for item in items], errors=errors,
                                   unchanged=unchanged)
    logger.info(f"设施底图坐标：{', '.join(item['facility_id'] for item in items)}，镜头注视点 "
                f"{np.round(gaze_position(report)).astype(int).tolist()}"
                + (f'，已有坐标偏差 {errors}' if errors else '') + '。')
    return dict(field=after, unchanged=unchanged)


def seek_map_point(runner, point, tag='facility_seek', pans=SEEK_PANS):
    """闭环把镜头移到底图点附近：每次重新配准展开地图，按局部方向拖动。

    返回 dict(client=目标预计客户区位置, field=新帧) 或 None（拖不动、配准失败或未确认小队
    位置）。只移动镜头，不发送走位。小队在寻位期间不动：看到箭头时记下它的底图坐标，之后
    每次配准都投影回画面，仍在画面附近就避开它，投影已在展开 ROI 外即确认离屏。
    实录 2026-10-22（22-36）：拖一次后小队离屏，离屏头像被小地图挡住，原先因此停止寻位。
    """
    localizer = map_localizer(runner)
    if localizer is None:
        return None
    chart, point, previous, squad = gaze_chart(localizer), np.asarray(point, float), None, None
    for index in range(pans + 1):
        try:
            report = locate_view(runner, f'{tag}_{index:02d}')
        except ValueError as exc:
            logger.info(f'寻找已知设施时镜头配准未通过：{exc}')
            return None
        field = runner.map_frame(track=False)
        client = map_to_client(chart, report, point)
        if client is not None and inside(client, SEEK_VIEW):
            return dict(client=client, field=field)
        distance = float(np.linalg.norm(gaze_position(report) - point))
        if index == pans or (previous is not None and distance > previous - SEEK_STUCK):
            logger.info(f'镜头未能移到已知设施附近（剩余底图距离 {distance:.0f}px）。')
            return None
        previous = distance
        arrow = squad_arrow(field)
        if arrow is not None:
            ground = np.asarray(arrow) + [0., goto.ARROW_GROUND_OFFSET]
            squad = client_to_map(chart, report, ground)
        elif squad is not None:
            # 投影不在展开 ROI 里（None）说明小队远在画面外；否则按投影位置避让。
            ground = map_to_client(chart, report, squad)
        elif squad_offscreen(field) is not None:
            ground = None
        else:
            logger.info('未确认小队位置或离屏标记，停止拖镜头以免误发走位。')
            return None
        try:
            drag = camera_drag(chart, report, point)
        except ValueError:
            return None
        runner.session.begin_target(tag)
        pan_scene(runner.session, drag, squad_ground=ground, settle=None, paced=True)
    return None


def remember_manual_entry(runner, catalog, stage):
    """需手动的备战（EX）关闭后小队就站在该敌人旁：记下小队底图坐标，换了镜头也认得它。

    实录 2026-10-22（22-36）：三个 EX 敌人每次关闭备战后换了镜头就是新编号，下一轮又被选中，
    来回走了两圈没有进展。展开地图里没看到小队时不记，保持原有行为。
    """
    if catalog is None:
        return None
    try:
        report = locate_view(runner, 'manual_entry')
    except ValueError as exc:
        logger.info(f'未能记下需手动关卡位置：{exc}')
        return None
    if report.get('position_kind') != 'squad':
        logger.info('展开地图里没看到小队，未记下需手动关卡位置。')
        return None
    point = np.asarray(report['position'], float)
    entries = catalog.data.setdefault('manual_entries', [])
    if not any(np.linalg.norm(np.asarray(entry['map_position']) - point) <= MANUAL_ENTRY_RADIUS
               for entry in entries):
        entries.append(dict(stage=list(stage), map_position=point.tolist(), at=time.time()))
        catalog.save()
    logger.info(f'记下需手动关卡 {list(stage)} 的底图位置 {np.round(point).astype(int).tolist()}，'
                '之后选目标时跳过这个位置的敌人。')
    return point


def manual_entry_check(runner, catalog, point, field):
    """目标落点是否就是已知需手动关卡；返回 'skip'、'changed'（开合地图后画面变了）或 None。

    只有本章记过需手动关卡时才配准一次展开地图；配准失败按 None 处理，照常尝试目标。
    """
    entries = [] if catalog is None else catalog.data.get('manual_entries', [])
    if not entries:
        return None
    try:
        report = locate_view(runner, 'manual_entry_check')
    except ValueError:
        return None
    where = client_to_map(gaze_chart(map_localizer(runner)), report, point)
    nearest = min(entries, key=lambda entry: np.linalg.norm(np.asarray(entry['map_position']) - where))
    distance = float(np.linalg.norm(np.asarray(nearest['map_position']) - where))
    if distance <= MANUAL_ENTRY_RADIUS:
        logger.info(f"目标落点在已知需手动关卡 {nearest['stage']} 旁（底图距离 {distance:.0f}px），跳过。")
        return 'skip'
    return None if scene_unchanged(field, runner.map_frame(track=False)) else 'changed'
