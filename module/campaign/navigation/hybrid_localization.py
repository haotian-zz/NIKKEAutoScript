"""原生寻路受阻时按需确认真实小队位置，不从地图视野中心构造小队坐标。"""

import time

import numpy as np

from module.campaign.vision.observation import EXPANDED_ROI
from module.logger import logger

# 相邻视野重叠，实际覆盖由每一帧观测决定；拖动增益和地图边界不保证等于这些计划偏移。
SEARCH_OFFSETS = ((0, 0), (240, 0), (240, 240), (0, 240), (-240, 240),
                  (-240, 0), (-240, -240), (0, -240), (240, -240))
CONFIRM_DISTANCE = 8.
SEARCH_SECONDS = 60.


def confirmed_pair(first, second):
    """仅接受同版底图中相邻两帧独立通过定位且相差不超过8地图像素的实测圆环。

    不向配准器提供期望坐标，不利用旧位置强制得到一致结果；缺标记、图版本变化和歧义均拒绝。
    返回值只是小队位置的一致性，不能证明道路可达、机关开启或传送完成。
    """
    if (first is None or second is None or first['status'] != 'accepted' or second['status'] != 'accepted'
            or first['identity'] != second['identity'] or not first['map_sha256']
            or first['map_sha256'] != second['map_sha256']
            or first.get('measured_ring') is None or second.get('measured_ring') is None):
        return False
    points = np.asarray([first.get('position'), second.get('position')], dtype=float)
    return (points.shape == (2, 2) and np.isfinite(points).all()
            and np.linalg.norm(points[0]-points[1]) <= CONFIRM_DISTANCE)


def position_feedback(before, after):
    """比较设施操作前后已确认位置，只报告位移，不将同区域移动认作传送或开门。

    只允许同章同难度同版底图；任一阶段未通过双帧确认时保留unknown。阈值沿用定位一致性
    容差，position_unchanged也只是未测得大于容差的位移，不能证明操作期间从未移动。
    """
    result = dict(state='unknown', transfer='unconfirmed', activated='unconfirmed')
    if (before['status'] != 'accepted' or after['status'] != 'accepted'
            or before['identity'] != after['identity'] or not before['map_sha256']
            or before['map_sha256'] != after['map_sha256']):
        return result
    distance = float(np.linalg.norm(np.asarray(before['position'])-after['position']))
    result.update(state='moved' if distance > CONFIRM_DISTANCE else 'position_unchanged',
                  distance_map_pixels=distance)
    return result


def locate_squad(session, tag='structure_localization'):
    """在正式会话内有界搜索小地图圆环，离开前收起面板，不发送任何场景落点。

    紧凑帧能定位时只补一帧确认；否则最多搜索九个展开视野，每个接受帧补一次独立确认。
    所有UI输入复用会话的焦点、停止、弹窗与输入预算闸门。60秒预算及视野数均固定，失败返回
    needs_review，不将地图覆盖不足解释为不通。清理仍遵守停止和焦点闸门，不在取消后强点控件。
    """
    owner, started = session.owner, time.monotonic()
    identity = list(session.expected_identity)
    result = dict(status='needs_review', identity=identity, map_sha256=None, position=None,
                  observations=[], scene_inputs=0, approved_routes=0, learned_transfers=0)
    session.begin_target(tag)

    def check():
        """先执行正式停止与窗口检查，内部到期只结束观察，不追加搜索手势。"""
        session.check()
        return time.monotonic()-started < SEARCH_SECONDS

    def observe():
        """只记录新鲜圆环和原PR接受结果，不运行整幅对象网格推理或保存base64面板。"""
        field = session.win.client()
        session.identity(field)
        snapshot = owner.observer.observe(field, tuple(identity), objects=False)
        matcher = owner.observer.matcher
        panel = snapshot.get('minimap') or {}
        row = dict(status=snapshot['localization'], identity=identity,
                   map_sha256=None if matcher is None else matcher.digest,
                   position=snapshot.get('squad'), mode=panel.get('mode'),
                   measured_ring=panel.get('squad'), road_iou=snapshot.get('road_iou'))
        result['observations'].append(row)
        session.record_evidence(tag, field, localization=row, arrays=dict(field=field))
        return row

    def confirm(first):
        """首帧通过后等待新鲜一帧独立复核，失败不沿用首帧许可。"""
        if first['status'] != 'accepted' or not check():
            return False
        session.pause(.3)
        second = observe()
        if not confirmed_pair(first, second):
            return False
        result.update(status='accepted', map_sha256=second['map_sha256'], position=second['position'])
        return True

    try:
        if not check():
            return result
        if confirm(observe()) or owner.observer.matcher is None:
            return result
        if not owner.expand_minimap(check=session.before_input, pause=session.pause):
            return result
        offset = (0, 0)
        left, top, right, bottom = EXPANDED_ROI
        center = np.array([(left+right)//2, (top+bottom)//2])
        for point in SEARCH_OFFSETS:
            if not check():
                break
            delta = np.asarray(point)-offset
            if np.any(delta):
                # 输入处理器收屏幕坐标；偏移仅用于地图面板内部拖动，不投影成场景走位。
                field = session.win.client()
                session.identity(field)
                if not owner.minimap_expanded(field):
                    raise RuntimeError('小队定位期间展开面板消失，未发送平移。')
                origin = np.asarray(session.win.client_origin())
                start, end = origin+center-delta//2, origin+center+delta//2
                if session.ui_validator is not None:
                    raise RuntimeError('其他UI核验仍在进行，不能开始地图搜索。')
                session.ui_validator = owner.minimap_expanded
                try:
                    session.win.handler.mouse_swipe(tuple(start.tolist()), tuple(end.tolist()))
                finally:
                    session.ui_validator = None
                session.pause(1.)
            offset = point
            if confirm(observe()):
                break
        return result
    finally:
        if not owner.collapse_minimap(check=session.before_input, pause=session.pause):
            raise RuntimeError('小队定位后无法收起面板，禁止继续场景输入。')
        result['seconds'] = time.monotonic()-started
        logger.info(f"受阻小队定位：{result['status']}，观测{len(result['observations'])}帧，"
                    f"位置{result['position']}；未批准道路或设施连接。")
