"""客户区与小地图里的目标识别：把 YOLO 类别统一成后续推图逻辑使用的语义对象。"""

import numpy as np

from module.campaign.vision.detection import (
    GRID_LABELS, LEGACY_SCENE_LABELS, MINIMAP_LABELS, Detection,
    detect_grid, detect_minimap, detect_scene, detector, suppress,
)

# 模型类别到语义的映射；state 区分同一类目标的状态或形态，后续逻辑按 kind 取、按 state 判断。
OBJECTS = {
    'scene_squad_arrow': ('squad_arrow', None),
    'scene_collectible_indicator': ('collectible_indicator', None),
    'scene_ground_mechanism_off': ('ground_mechanism', 'off'),
    'scene_ground_mechanism_on': ('ground_mechanism', 'on'),
    'scene_elevator_start': ('elevator', 'start'),
    'scene_elevator_end': ('elevator', 'end'),
    'scene_barrier_curtain': ('barrier', 'curtain'),
    'scene_barrier_pillar': ('barrier', 'pillar'),
    'scene_enemy_ring': ('enemy_ring', None),
    'minimap_enemy_normal': ('enemy', 'normal'),
    'minimap_enemy_ex': ('enemy', 'ex'),
    'minimap_squad_ring': ('squad_ring', None),
}
# 整幅画面网格检测的语义类别，也是 detect_objects 的默认范围。
GRID_KINDS = ('ground_mechanism', 'elevator', 'barrier', 'enemy_ring')
SCENE_KINDS = GRID_KINDS + ('squad_arrow', 'collectible_indicator')
MINIMAP_KINDS = ('enemy', 'squad_ring')


def requested(kinds, default):
    """统一校验语义类别参数；单个字符串会被逐字符当成类别，必须先挡掉。"""
    if kinds is None:
        return default
    if isinstance(kinds, str):
        raise ValueError('kinds 需要语义类别的序列，而不是单个字符串。')
    return tuple(kinds)


def kind_labels(kinds, available):
    """把语义类别换成模型类别；未知类别或取图方式不匹配时报错，不静默忽略。"""
    names = {OBJECTS[label][0] for label in available}
    selected = requested(kinds, ())
    unknown = [kind for kind in selected if kind not in names]
    if unknown:
        raise ValueError(f'这些目标类别不能用当前取图方式检测：{", ".join(map(str, unknown))}。')
    return tuple(label for label in available if OBJECTS[label][0] in selected)


def described(detection, space):
    """把单个模型检测结果整理成后续逻辑使用的语义对象。

    space 记录坐标系（client 为客户区像素、minimap 为面板像素），两者不能混用；坐标保留一位
    小数、置信度保留四位，避免日志和快照里出现无意义的长浮点数。
    """
    kind, state = OBJECTS[detection.label]
    return dict(label=detection.label, kind=kind, state=state, confidence=round(detection.confidence, 4),
                position=[round(float(value), 1) for value in detection.center],
                box=[round(float(value), 1) for value in detection.box], space=space)


def detect_objects(image, kinds=None):
    """1776×999 BGR 客户区截图里的场景目标，坐标为客户区像素。

    默认检测机关、电梯、屏障和敌人脚圈，整幅画面切八片推理。脚圈不直接并入敌人记忆。
    小队箭头和收集提示另有已验证的遮罩切片取图方式，只有显式请求时才多跑一遍；
    箭头的精确中心仍应使用 perception.squad_arrow。
    """
    selected = requested(kinds, GRID_KINDS)
    grid = kind_labels([kind for kind in selected if kind in GRID_KINDS], GRID_LABELS)
    legacy = kind_labels([kind for kind in selected if kind not in GRID_KINDS], LEGACY_SCENE_LABELS)
    detections = list(detect_grid(image, grid)) if grid else []
    if legacy:
        detections += detect_scene(image, legacy)
    return sorted((described(item, 'client') for item in detections),
                  key=lambda item: item['confidence'], reverse=True)


def nearby_platforms(image, ground):
    """在可信小队周围分级补查400px平台裁图，保留原模型分数与客户区坐标。

    用于原生寻路停滞后的设施诊断，补足640网格对近端小平台的尺度/上下文漏检。
    中央一片没有完整高分平台时才扩展四个邻域，最多五片；裁图保持400×400并夹回客户区。
    不移动游戏镜头、不修改全局阈值、不推测可达或配对，仍须下游两帧及0.9交互门槛。
    """
    ground = np.asarray(ground, float)
    if image.shape != (999, 1776, 3) or ground.shape != (2,) or not np.isfinite(ground).all():
        raise ValueError('近端平台补查要求正式客户区和可信二维小队落点。')
    from module.campaign.vision.enemy_ring import clickable_ring
    found, seen = [], set()
    for index, offset in enumerate(((0, 0), (-300, 0), (300, 0), (0, -240), (0, 240))):
        left, top = np.clip(np.rint(ground+offset-200), (0, 0), (1376, 599)).astype(int)
        if (left, top) in seen:
            continue
        seen.add((left, top))
        local = detector().predict(image[top:top+400, left:left+400],
                                   ('scene_elevator_start', 'scene_elevator_end'))
        shifted = [Detection(item.label, item.confidence, tuple(np.asarray(item.box)+[left, top, left, top]))
                   for item in local]
        found.extend(item for item in shifted if clickable_ring(item.box))
        if index == 0 and any(item.confidence >= .9 for item in found):
            break
    return [dict(described(item, 'client'), view='squad_local_400') for item in suppress(found)]


def visible_platforms(image):
    """缺少小队锚点时以400px网格补查平台，不生成小队位置或推断平台可达。

    仅供稳定任务标记下、无法安全拖镜头的有界恢复。15片覆盖正式客户区；最多把两个
    最高分提案重新居中补查，匹配框IoU须≥0.5。保留原类别阈值，输入前仍须两帧同状态
    且≥0.9、唯一可交互平台；不能把提案存在当作允许点击或传送成功。
    """
    if image.shape != (999, 1776, 3) or image.dtype != np.uint8:
        raise ValueError('无小队锚点的平台补查要求1776×999 uint8客户区。')
    from module.campaign.vision.enemy_ring import clickable_ring
    model, found = detector(), []
    labels = ('scene_elevator_start', 'scene_elevator_end')
    for top in (0, 300, 599):
        for left in (0, 344, 688, 1032, 1376):
            for item in model.predict(image[top:top+400, left:left+400], labels):
                shifted = Detection(item.label, item.confidence, tuple(np.asarray(item.box)+[left, top, left, top]))
                if clickable_ring(shifted.box):
                    found.append(shifted)
    proposals = sorted(suppress(found), key=lambda item: item.confidence, reverse=True)
    for proposal in proposals[:2]:
        left, top = np.clip(np.rint(proposal.center-200), (0, 0), (1376, 599)).astype(int)
        for item in model.predict(image[top:top+400, left:left+400], labels):
            shifted = Detection(item.label, item.confidence, tuple(np.asarray(item.box)+[left, top, left, top]))
            a, b = np.asarray(proposal.box), np.asarray(shifted.box)
            intersection = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
            union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
            if intersection/max(1., union) >= .5 and clickable_ring(shifted.box):
                found.append(shifted)
    return [dict(described(item, 'client'), view='scene_platform_400') for item in suppress(found)]


def detect_minimap_objects(panel, kinds=None):
    """小地图面板里的小队圆环与敌人标记，坐标为面板像素。

    panel 由调用方按展开或紧凑模式裁好；圆环与普通敌人的精确中心仍应使用 perception 中
    对应的精定位，这里只给出检测框中心。
    """
    labels = kind_labels(requested(kinds, MINIMAP_KINDS), MINIMAP_LABELS)
    return sorted((described(item, 'minimap') for item in detect_minimap(panel, labels)),
                  key=lambda item: item['confidence'], reverse=True)
