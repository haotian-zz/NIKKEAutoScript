"""临时移除紧凑小地图遮挡，读取真实离屏头像；不点击头像或生成小队地面位置。"""

import cv2
import numpy as np

from module.campaign.vision.minimap import map_visible, minimap_control_visible
from module.campaign.vision.offscreen import squad_offscreen


def compact_close_point(field):
    """同时确认紧凑地图、左上控件及右上白色X，返回实测控件中心，不提供固定落点兜底。

    只接受1776×999客户区；在紧凑面板已知按钮区域匹配17～23px的X，相关分数至少0.7。
    调用者须双帧复核并通过正式UI输入前校验；此点只用于临时收起遮挡，不是场景目标。
    """
    if (field.shape != (999, 1776, 3) or not map_visible(field[118:273, 27:207])
            or not minimap_control_visible(field[76:120, 20:64])):
        return None
    hsv = cv2.cvtColor(field[80:121, 180:229], cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 210), (179, 80, 255))
    best = None
    for size in (17, 19, 21, 23):
        for thickness in (3, 4, 5):
            template = np.zeros((size, size), np.uint8)
            cv2.line(template, (1, 1), (size-2, size-2), 255, thickness)
            cv2.line(template, (1, size-2), (size-2, 1), 255, thickness)
            _, score, _, point = cv2.minMaxLoc(cv2.matchTemplate(white, template, cv2.TM_CCOEFF_NORMED))
            if score >= .7 and (best is None or score > best[0]):
                best = score, [180+point[0]+size/2, 80+point[1]+size/2]
    return None if best is None else best[1]


def unoccluded_offscreen(session):
    """仅在双帧确认X后临时收起小地图，以原头像规则连续复核，正常返回前重新打开地图。

    每次最多一次收起和一次恢复，不拖镜头，不点击头像，不降低头像规则。取消及输入前控件
    变化均保留正式闸门；恢复失败直接报错，禁止在未知地图状态继续行动。
    """
    field = session.win.client()
    session.identity(field)
    already_closed = session.owner.observer.minimap_closed(field)
    point = None
    if not already_closed:
        first = compact_close_point(field)
        if first is None:
            return None
        session.pause(.25)
        fresh = session.win.client()
        session.identity(fresh)
        point = compact_close_point(fresh)
        if point is None or np.linalg.norm(np.asarray(point)-first) > 3.:
            return None

    def verified(current):
        """在真正发送输入前重新核验同一个X，避免面板消失后落到场景。"""
        actual = compact_close_point(current)
        return actual is not None and np.linalg.norm(np.asarray(actual)-point) <= 3.

    try:
        if not already_closed:
            session.tap_verified_ui(point, verified)
            session.pause(.7)
        field = session.win.client()
        session.identity(field)
        if not session.owner.observer.minimap_closed(field):
            raise RuntimeError('小地图遮挡补查未确认收起，未继续场景输入。')
        first_marker = squad_offscreen(field)
        session.record_evidence('offscreen_unoccluded', field, marker=first_marker, arrays=dict(field=field))
        session.pause(.3)
        fresh = session.win.client()
        session.identity(fresh)
        if not session.owner.observer.minimap_closed(fresh):
            raise RuntimeError('头像复核期间小地图状态变化，未继续场景输入。')
        marker = squad_offscreen(fresh)
        session.record_evidence('offscreen_unoccluded_confirm', fresh, marker=marker, arrays=dict(field=fresh))
        if (first_marker is None or marker is None
                or np.linalg.norm(np.asarray(marker['center'])-first_marker['center']) > 3.):
            return None
        return marker
    finally:
        session.owner.device.screenshot()
        current = session.win.client()
        if session.owner.observer.minimap_closed(current):
            restored = session.owner.reopen_minimap_if_closed(current, check=session.before_input, pause=session.pause)
            if not restored:
                raise RuntimeError('头像补查后小地图未恢复，停止后续行动。')
