"""用独立小地图标记构造局部场景标定候选；候选必须再经真实移动校验。"""

from itertools import permutations

import cv2
import numpy as np


def stable_scene_triple(first, second):
    """连续两帧仅有三个高分普通敌人脚圈且唯一对应时，返回第二帧真实客户区坐标。

    圈分数至少0.85，相邻帧距离≤8像素，同帧任意两个圈距离至少30像素。
    不用单帧漏检挑出有利的三个；没有三点唯一关系时禁止构造标定。
    """
    rows = [[item for item in values if item['kind'] == 'enemy_ring' and item['confidence'] >= .85]
            for values in (first, second)]
    if any(len(values) != 3 for values in rows):
        raise ValueError('场景需要双帧恰好三个高分脚圈。')
    points = [np.asarray([item['position'] for item in values], float) for values in rows]
    if any(min(np.linalg.norm(p[i]-p[j]) for i in range(3) for j in range(i)) < 30 for p in points):
        raise ValueError('场景脚圈存在近邻竞争。')
    distances = np.linalg.norm(points[1][:, None]-points[0][None], axis=2)
    matches = distances <= 8.
    if not (np.all(matches.sum(axis=0) == 1) and np.all(matches.sum(axis=1) == 1)):
        raise ValueError('场景脚圈双帧不一致。')
    return points[1]


def stable_map_landmarks(document):
    """从零场景输入的面板巡检提取跨视野一致的三个普通敌人，保留同版地图约束。

    每个点至少在两个视野各出现两帧，置信度不低于0.9，组内最大距离12地图像素。
    viewport配准只提供敌人的地图坐标，绝不将其position当小队。当前原型仅处理三个
    互相至少30像素远的普通敌人；数量或关联不明确时拒绝，不从部分组合挑有利解。
    """
    location = document['navigation_decision']['location']
    if document['scene_inputs'] or location['status'] != 'accepted' or not location['map_sha256']:
        raise ValueError('标记来源必须零场景输入且具有独立确认的小队定位。')
    groups = []
    for row in document['expanded_correspondences']['samples']:
        if row['status'] != 'accepted':
            continue
        registration = row['registration']
        if registration['image_sha256'] != location['map_sha256']:
            raise ValueError('地图标记来自不同底图。')
        for item in row['enemies']:
            if item['kind'] != 'normal' or item['confidence'] < .9:
                continue
            point = np.asarray(item['map_position'], float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError('地图敌人坐标无效。')
            near = [group for group in groups if np.linalg.norm(point-np.mean(group['points'], axis=0)) <= 12.]
            if len(near) > 1:
                raise ValueError('地图标记存在竞争关联。')
            if not near:
                groups.append(dict(points=[], frames=[]))
                near = groups[-1:]
            group = near[0]
            frame = (row['view'], row['sample'])
            if frame in group['frames']:
                raise ValueError('一帧中多个标记竞争同一身份。')
            group['points'].append(point)
            group['frames'].append(frame)
    if len(groups) != 3:
        raise ValueError('当前局部标定需要恰好三个唯一普通敌人。')
    centers = []
    for group in groups:
        points = np.asarray(group['points'])
        views = {view for view, _ in group['frames'] if sum(v == view for v, _ in group['frames']) >= 2}
        if len(views) < 2 or np.max(np.linalg.norm(points[:, None]-points[None], axis=2)) > 12.:
            raise ValueError('地图标记缺少跨视野双帧一致性。')
        centers.append(points.mean(axis=0))
    centers = np.asarray(centers)
    if min(np.linalg.norm(centers[i]-centers[j]) for i in range(3) for j in range(i)) < 30.:
        raise ValueError('地图标记过近，无法可靠区分。')
    return dict(map_sha256=location['map_sha256'], positions=centers.tolist(),
                observations=[dict(frames=group['frames'], points=np.asarray(group['points']).tolist())
                              for group in groups])


def local_candidate(landmarks, scene_points, squad_position, ground):
    """以三角布局关联三个敌人，再加实测小队构造局部单应；结果始终未验证。

    客户区与底图都采用右／下正方向，禁止镜像关联。三个敌人的最优相似匹配RMS须≤12
    地图像素，次选至少差10像素；该匹配只用于身份，不把相似变换当作地面投影。
    四点单应没有独立精度证明，返回unvalidated，调用者只能显式做一次校准试验。
    """
    scene = np.asarray(scene_points, float)
    target = np.asarray(landmarks['positions'], float)
    squad, ground = np.asarray(squad_position, float), np.asarray(ground, float)
    if (scene.shape != (3, 2) or target.shape != (3, 2) or squad.shape != (2,) or ground.shape != (2,)
            or not all(np.isfinite(value).all() for value in (scene, target, squad, ground))):
        raise ValueError('局部标定的实测对应点不完整。')
    centered = scene-scene.mean(axis=0)
    denominator = float((centered**2).sum())
    if denominator < 1000 or abs(np.cross(scene[1]-scene[0], scene[2]-scene[0])) < 1000:
        raise ValueError('场景三角布局退化。')
    candidates = []
    for order in permutations(range(3)):
        paired = target[list(order)]
        offset = paired-paired.mean(axis=0)
        u, values, vt = np.linalg.svd(centered.T @ offset)
        rotation = u @ vt
        if np.linalg.det(rotation) <= 0:
            continue
        scale = float(values.sum()/denominator)
        error = float(np.sqrt(np.mean(np.sum((scale*centered @ rotation-offset)**2, axis=1))))
        candidates.append(dict(order=order, error=error, scale=scale))
    candidates.sort(key=lambda item: item['error'])
    if (len(candidates) < 2 or candidates[0]['error'] > 12.
            or candidates[1]['error']-candidates[0]['error'] < 10.):
        raise ValueError('场景与地图敌人身份仍有歧义。')
    paired = target[list(candidates[0]['order'])]
    source = np.vstack((ground, scene)).astype(np.float32)
    destination = np.vstack((squad, paired)).astype(np.float32)
    hull = cv2.convexHull(source)
    if len(hull) != 4 or cv2.contourArea(hull) < 3000:
        raise ValueError('四点支撑区域退化，未构造标定。')
    matrix = cv2.getPerspectiveTransform(source, destination)
    depth = np.column_stack((source, np.ones(4))) @ matrix[2]
    if (not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10
            or not (np.all(depth > .05) or np.all(depth < -.05))):
        raise ValueError('局部投影奇异或穿过地平线。')
    return dict(status='unvalidated', map_sha256=landmarks['map_sha256'], matrix=matrix.tolist(),
                source_points=source.tolist(), map_points=destination.tolist(), support=hull[:, 0].tolist(),
                association=candidates, independent_motion_points=0, approved_routes=0)


def calibration_point(candidate, pillars, road):
    """在实测四点凸包内部选一次短移，留在两个光幕立柱同侧，不越过设施或敌人。

    仅为显式校准实验生成候选，不授予普通执行器通行许可。距离实测小队60～120客户区
    像素、距支撑边至少2像素、距光幕线至少25像素；地图候选须落在已知道路上。
    不用颜色猜道路，不用画面中心生成小队，不向未知区域外推。
    """
    if candidate['status'] != 'unvalidated':
        raise ValueError('校准短移只接受明确的未验证候选。')
    source = np.asarray(candidate['source_points'], float)
    ground = source[0]
    hull = np.asarray(candidate['support'], np.float32).reshape(-1, 1, 2)
    pillars = np.asarray(pillars, float)
    if pillars.shape != (2, 2) or not np.isfinite(pillars).all():
        raise ValueError('短移需要两根实际光幕立柱限制同侧。')
    axis = pillars[1]-pillars[0]
    length = float(np.linalg.norm(axis))
    if length < 40:
        raise ValueError('光幕立柱间距不足。')
    home = float(np.cross(axis, ground-pillars[0])/length)
    direction = source[1:].mean(axis=0)-ground
    direction /= np.linalg.norm(direction)
    matrix = np.asarray(candidate['matrix'])
    for distance in range(120, 59, -5):
        point = ground+direction*distance
        side = float(np.cross(axis, point-pillars[0])/length)
        if (abs(side) < 25 or home*side <= 0
                or cv2.pointPolygonTest(hull, tuple(point), True) < 2
                or np.min(np.linalg.norm(source[1:]-point, axis=1)) < 150):
            continue
        mapped = matrix @ np.r_[point, 1.]
        mapped = mapped[:2]/mapped[2]
        x, y = np.rint(mapped).astype(int)
        if not (0 <= y < road.shape[0] and 0 <= x < road.shape[1] and road[y, x]):
            continue
        return dict(click=point.tolist(), predicted_map=mapped.tolist(), distance_client=float(distance),
                    purpose='one_explicit_calibration_step', action_approved=False)
    raise ValueError('没有同时满足支撑、光幕同侧和已知道路的校准落点。')
