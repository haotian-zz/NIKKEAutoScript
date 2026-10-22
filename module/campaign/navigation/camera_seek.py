"""复用PR方向计算与面板定位寻找设施候选区域；只拖镜头，不发送移动点击。"""

import time

import numpy as np


class CameraSeekUnconfirmed(RuntimeError):
    """地图区域、镜头响应或真实小队避让未确认；终止本次寻位，不继续猜方向。"""


def validate_hint(hint, identity, digest, size):
    """校验外部线索绑定的章节、底图及范围，只返回观察区域，不授予设施交互许可。

    视频近似标记可作摄像机区域线索，不能当作精确落点。目标使用底图像素，半径30～250。
    线索必须显式标明camera_region_only，不能将历史小队位置伪装成新鲜行动坐标。
    """
    point = np.asarray(hint['center_map'], float)
    radius = hint.get('radius_map', 180.)
    if (hint.get('purpose') != 'camera_region_only' or hint.get('identity') != list(identity)
            or not digest or hint.get('map_sha256') != digest
            or point.shape != (2,) or not np.isfinite(point).all()
            or np.any(point < 0) or np.any(point >= np.asarray(size))
            or type(radius) not in (int, float) or not 30. <= radius <= 250.):
        raise ValueError('区域线索未绑定当前身份、底图或有效范围。')
    return point, float(radius)


def seek_region(session, hint, max_pans=8, progress=None, body_reader=None, catalog=None):
    """逐次可靠定位镜头，按PR计算粗方向寻找外部区域，每次最多八次受保护场景拖动。

    两帧面板投影中心须相差≤8地图像素；无真实圆环时只保留viewport类型。原PR标定在
    此处仅估算拖镜头方向，不生成小队位置、设施地图坐标或地面点击。每次拖动后重新
    定位并要求接近目标；没有实测箭头、真实离屏证据或显式核验人物框时不拖。人物框仅保护按下点。总时限120秒，不续期。
    """
    from module.campaign.navigation import controls as goto
    from module.campaign.navigation.camera_navigation import ROI_BOUNDS, camera_drag, inside, pan_scene
    from module.campaign.session import CampaignLocalizer
    from module.campaign.vision.observation import EXPANDED_ROI
    from module.campaign.vision.minimap_occlusion import unoccluded_offscreen
    from module.campaign.vision.offscreen import squad_offscreen
    from module.campaign.vision.perception import squad_arrow
    from module.campaign.vision.surface_motion import project
    if type(max_pans) is not int or not 1 <= max_pans <= 8:
        raise ValueError('区域寻位最多八次镜头动作。')
    owner = session.owner
    chart = owner.ensure_chart(session.expected_identity[0])
    if chart is None or owner.observer.matcher is None:
        raise CameraSeekUnconfirmed('区域寻位缺少当前底图。')
    target, radius = validate_hint(hint, session.expected_identity, owner.observer.matcher.digest, chart.size)
    localizer = CampaignLocalizer(owner.observer.matcher, owner.cut)
    affine = np.eye(3)
    affine[:2, :2] = goto.A_INV
    direction_chart = affine @ localizer.old_matrix
    result = dict(state='observing', scene_inputs=0, approved_routes=0, learned_transfers=0,
                  target_map=target.tolist(), radius_map=radius, observations=[], camera_pans=0)
    session.begin_target('camera_region_seek')
    if catalog is not None and not catalog.track_background(session.win.client(), force=True):
        raise CameraSeekUnconfirmed('寻位起始视野未关联设施关键帧，未发送镜头拖动。')
    started, previous_distance = time.monotonic(), None

    def publish():
        """仅输出小型数值进度；镜头运动和观察成功不记作通行进展。"""
        if progress is not None:
            progress(result)

    def observe(index):
        """通过原PR定位器双帧读取当前地图视野，离开前恢复紧凑面板。"""
        if not owner.expand_minimap(check=session.before_input, pause=session.pause):
            raise CameraSeekUnconfirmed('区域寻位无法展开地图。')
        try:
            reports, markers = [], []
            for sample in range(2):
                session.check()
                field = session.win.client()
                session.identity(field)
                if not owner.minimap_expanded(field):
                    raise CameraSeekUnconfirmed('寻位观测中展开面板消失。')
                left, top, right, bottom = EXPANDED_ROI
                panel = field[top:bottom, left:right]
                try:
                    reports.append(localizer.locate(panel, f'camera_seek_{index}_{sample}', require_player=False))
                except (RuntimeError, ValueError) as error:
                    session.record_evidence('camera_region_localization_rejected', field, error=str(error),
                                            index=index, sample=sample, arrays=dict(field=field, panel=panel))
                    raise
                markers.append(squad_offscreen(field))
                session.pause(.3)
            centers = [project(report['roi_to_map'], [243., 231.]) for report in reports]
            if np.linalg.norm(centers[0]-centers[1]) > 8.:
                raise CameraSeekUnconfirmed('镜头双帧位置不一致。')
            marker = (markers[-1] if all(item is not None for item in markers)
                      and np.linalg.norm(np.asarray(markers[0]['center'])-markers[1]['center']) <= 3. else None)
            row = dict(index=index, center_map=centers[-1].tolist(), expanded_offscreen_marker=marker,
                       position_kind=reports[-1]['position_kind'], reports=reports)
            result['observations'].append(row)
            captured = session.win.client()
            session.record_evidence('camera_region_observation', captured, result=row, arrays=dict(field=captured))
            publish()
            return reports[-1], centers[-1], marker
        finally:
            if not owner.collapse_minimap(check=session.before_input, pause=session.pause):
                raise CameraSeekUnconfirmed('区域寻位后地图未恢复紧凑状态。')

    for index in range(max_pans+1):
        session.check()
        if time.monotonic()-started > 120.:
            raise CameraSeekUnconfirmed('区域寻位超过120秒预算。')
        view, center, observed_marker = observe(index)
        distance = float(np.linalg.norm(target-center))
        result['distance_map'] = distance
        target_roi = project(np.linalg.inv(view['roi_to_map']), target)
        result['target_roi'] = target_roi.tolist()
        if distance <= radius or inside(target_roi, ROI_BOUNDS):
            result.update(state='region_observed', seconds=time.monotonic()-started)
            publish()
            return result
        if previous_distance is not None and distance >= previous_distance-5.:
            raise CameraSeekUnconfirmed('镜头未接近线索区域，停止本轮方向估计。')
        if index == max_pans:
            raise CameraSeekUnconfirmed('区域寻位镜头预算已用尽。')
        previous_distance = distance
        field = session.win.client()
        session.identity(field)
        arrow = squad_arrow(field)
        ground = None if arrow is None else np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET]
        body = None
        if ground is None and body_reader is not None:
            body = body_reader(field, view)
        if ground is None and body is None:
            # 展开态已移除左上遮挡；收起面板未移动镜头或小队，可复用刚才的双帧实际头像。
            marker = squad_offscreen(field) or observed_marker
            if marker is None:
                marker = unoccluded_offscreen(session)
            if marker is None:
                raise CameraSeekUnconfirmed('未确认小队或离屏头像，未发送镜头拖动。')
        drag = camera_drag(direction_chart, view, target)
        result['next_drag_client'] = drag.tolist()
        publish()
        pan_scene(session, drag, squad_ground=ground, squad_box=body)
        result['camera_pans'] += 1
        publish()
        if catalog is not None and not catalog.track_background(session.win.client(), force=True):
            raise CameraSeekUnconfirmed('镜头移动后设施背景未连续关联，停止寻位并保留旧尝试。')
    raise AssertionError('区域寻位未返回有效状态。')
