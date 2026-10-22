"""大型入口的骷髅语义与发光脚圈联合核验，不依赖地面主色或章节固定坐标。"""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from module.campaign.vision.detection import Detection
from module.campaign.vision.enemy_ring import HUD_RECTS, clickable_ring


def normalized_glyph(mask):
    """去掉二值字形外围空白后归一到64×48；空图返回None，保留眼窝等内部孔洞。"""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return cv2.resize(mask[ys.min():ys.max()+1, xs.min():xs.max()+1], (64, 48),
                      interpolation=cv2.INTER_AREA) > 100


@lru_cache(maxsize=1)
def skull_templates():
    """缓存固定骷髅UI的有限仿射变体，覆盖透视剪切及左右镜头外观，不学习地貌纹理。

    模板仅含41×31二值图标；剪切±0.6、旋转±10度及镜像共78种，进程内只生成一次。
    """
    source = cv2.imread(str(Path(__file__).parent / 'assets/boss_skull_mask.png'), cv2.IMREAD_GRAYSCALE)
    if source is None or not np.any(source):
        raise ValueError('缺少有效的Boss骷髅二值模板。')
    result = []
    for flip in (False, True):
        for shear in np.linspace(-.6, .6, 13):
            for angle in (-10, 0, 10):
                padded = np.pad(source, ((40, 40), (40, 40)))
                if flip:
                    padded = padded[:, ::-1]
                matrix = cv2.getRotationMatrix2D((padded.shape[1]/2, padded.shape[0]/2), angle, 1.)
                matrix[1, 0] += shear
                result.append(normalized_glyph(cv2.warpAffine(
                    padded, matrix, padded.shape[::-1], flags=cv2.INTER_NEAREST)))
    return np.asarray(result)


def skull_candidates(field):
    """从固定UI外的白色连通字形中找骷髅，返回框及最高仿射IoU，门槛为0.65。

    模板必须保留内部孔洞，孤立亮三角、橙圈和角色亮点不能凭亮度得到入口身份。
    分数是形状IoU而非校准概率；此阶段无点击许可，还须独立脚圈和连续帧验证。
    """
    hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (179, 100, 255))
    for left, top, right, bottom in HUD_RECTS:
        white[top:bottom, left:right] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(white)
    bank, result = skull_templates(), []
    for x, y, width, height, area in stats[1:]:
        if not (20 <= width <= 110 and 12 <= height <= 85 and 1.1 <= width/height <= 2.5 and area >= 80):
            continue
        glyph = normalized_glyph(white[y:y+height, x:x+width])
        scores = (glyph & bank).sum((1, 2)) / np.maximum(1, (glyph | bank).sum((1, 2)))
        score = float(scores.max())
        if score >= .65:
            result.append(dict(glyph=(int(x), int(y), int(width), int(height)), shape_score=score))
    return sorted(result, key=lambda item: -item['shape_score'])[:8]


def portal_below(red, glyph, spans=(2.8, 3.2, 3.6, 4., 4.4, 4.8), below=(1.4, 2.8), min_score=.2,
                 thickness=.12):
    """在图标下方的局部区域搜索完整椭圆环，返回实际匹配框与几何分数或None。

    搜索尺度随图标宽度变化：默认（骷髅）环宽2.8～4.8倍、宽高比0.45～0.9、中心在图标下方
    1.4～2.8倍宽度内。零均值相关至少min_score，拒绝均匀红地面；不按偏移直接生成落点。
    """
    x, y, width, height = glyph
    cx, cy = x+width/2, y+height/2
    reach = max(spans)/2+1.5
    left, right = max(0, round(cx-reach*width)), min(red.shape[1], round(cx+reach*width))
    top, bottom = round(cy+(below[0]-max(spans)*.45)*width), min(red.shape[0], round(cy+(below[1]+max(spans)*.45)*width))
    top = max(0, top)
    patch, best = red[top:bottom, left:right], None
    for span in spans:
        for ratio in (.45, .6, .75, .9):
            w, h = round(span*width), round(span*width*ratio)
            template = np.zeros((h+12, w+12), np.uint8)
            cv2.ellipse(template, (w//2+6, h//2+6), (w//2, h//2),
                        0, 0, 360, 255, max(3, round(width*thickness)))
            if template.shape[0] > patch.shape[0] or template.shape[1] > patch.shape[1]:
                continue
            scores = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            yy, xx = np.indices(scores.shape)
            centers_x, centers_y = left+xx+6+w/2, top+yy+6+h/2
            scores[(abs(centers_x-cx) > width) | (centers_y-cy < below[0]*width)
                   | (centers_y-cy > below[1]*width)] = -1
            _, score, _, point = cv2.minMaxLoc(scores)
            box = (left+point[0]+6, top+point[1]+6, left+point[0]+6+w, top+point[1]+6+h)
            if score >= min_score and clickable_ring(box) and (best is None or score > best['geometry']):
                best = dict(box=box, geometry=float(score))
    return best


def red_mask(field, hue_low=23):
    """入口自带的红色发光图形；不假设地面主色。hue_low 收窄可排除橙黄色楼沿。"""
    hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
    return (((hsv[:, :, 0] < hue_low) | (hsv[:, :, 0] > 160))
            & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 180)).astype(np.uint8)*255


# 主线菱形离左右边缘至少这么远才找其下方入口：贴边的是离屏指示，下面常是红色 EX 指示。
QUEST_PORTAL_EDGE = 200.
QUEST_PORTAL_SCORE = .3
# 主线入口光圈偏粉红；橙黄色楼顶边沿（色相 10～23）曾在 0.27 被误认。
QUEST_PORTAL_HUE = 10


def quest_portal_candidates(field):
    """画面内主线任务菱形正下方的完整红色入口环；返回 dict(marker, box, geometry) 列表。

    实录 2026-10-22（22-36）：主线关卡入口是菱形正下方约1.6～3.4倍菱形宽处的红色光圈，
    检测模型完全不出框，程序因此转去全章清怪巡检空转；用户确认点光圈即可进入。离线核对
    第 22 章 31 帧：真入口 0.32～0.40，贴边离屏菱形下的 EX 指示约 0.30（已按边距排除），橙黄楼沿 0.27（已按色相排除）。
    只提供候选，点击后仍由备战弹窗核对实际关卡。
    """
    from module.campaign.vision.story_marker import story_markers
    if field.shape != (999, 1776, 3):
        raise ValueError('主线入口核验要求1776×999客户区。')
    markers = [item for item in story_markers(field) if item['kind'] in ('scene', 'cue')
               and QUEST_PORTAL_EDGE <= item['position'][0] <= field.shape[1]-QUEST_PORTAL_EDGE]
    if not markers:
        return []
    red, result = red_mask(field, QUEST_PORTAL_HUE), []
    for marker in markers:
        x1, y1, x2, y2 = marker['box']
        ring = portal_below(red, (x1, y1, x2-x1, y2-y1), spans=(1.3, 1.5, 1.7, 1.9, 2.1, 2.4),
                            below=(1.6, 3.4), min_score=QUEST_PORTAL_SCORE, thickness=.1)
        if ring is not None:
            result.append(dict(marker=marker['position'], **ring))
    return result


def boss_ring_candidates(field):
    """返回同一帧同时具有骷髅与脚圈证据的候选；不发送输入或宣称实际关卡编号。

    只支持正式1776×999客户区。红色掩码仅测入口自带图形，不假设道路、沙漠或城市的颜色。
    无骷髅时不运行脚圈搜索；调用方仍须两帧稳定以及备战关卡校验。
    """
    if field.shape != (999, 1776, 3):
        raise ValueError('Boss入口核验要求1776×999客户区。')
    glyphs = skull_candidates(field)
    if not glyphs:
        return []
    red = red_mask(field)
    result = []
    for glyph in glyphs:
        ring = portal_below(red, glyph['glyph'])
        if ring is not None:
            result.append(dict(glyph, **ring))
    return result


def scene_boss_rings(field):
    """将双证据入口适配为场景脚圈；confidence保留字形IoU，不冒充模型概率或胜利证据。"""
    return [Detection('scene_enemy_ring', item['shape_score'], tuple(map(float, item['box'])))
            for item in boss_ring_candidates(field)]
