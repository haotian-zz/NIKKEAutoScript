"""EX 场景脚圈的补充识别：完整文字与橙色椭圆同时成立，避免模型漏检导致漏怪。"""

import cv2
import numpy as np

from module.campaign.vision.detection import Detection
from module.campaign.vision.enemy_ring import HUD_RECTS, clickable_ring


def ex_ring_candidates(field):
    """从白色短标题下方搜索橙色椭圆，返回待 OCR 复核的文字框、脚圈框与几何分数。

    不固定章节或画面坐标。字高 10～36px、标题宽高比 1.3～2.8，椭圆宽为字高 6～10 倍；
    相关分数至少 0.4 只代表候选，必须另有 EX 文字及两帧稳定证据才可发输入。
    """
    hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 210), (179, 100, 255))
    # E、X 是独立笔画；只横向连通最多两像素间隙，后续仍须完整文字和脚圈双重验证。
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8))
    for left, top, right, bottom in HUD_RECTS:
        white[top:bottom, left:right] = 0
    orange = cv2.inRange(hsv, (0, 45, 220), (30, 255, 255))
    _, _, components, _ = cv2.connectedComponentsWithStats(white)
    candidates = []
    for x, y, width, height, area in components[1:]:
        if not (10 <= height <= 36 and 1.3 <= width / height <= 2.8 and area >= 30):
            continue
        cx, cy = x + width / 2, y + height / 2
        left, right = max(0, int(cx - 6 * height)), min(field.shape[1], int(cx + 6 * height))
        top, bottom = int(cy + 2 * height), min(field.shape[0], int(cy + 10 * height))
        patch = orange[top:bottom, left:right]
        best = None
        for span in (6, 7, 8, 9, 10):
            for ratio in (.35, .45, .55, .65):
                w, h = round(span * height), round(span * height * ratio)
                template = np.zeros((h + 8, w + 8), np.uint8)
                cv2.ellipse(template, (w // 2 + 4, h // 2 + 4), (w // 2, h // 2),
                            0, 0, 360, 255, max(2, round(height * .12)))
                if template.shape[0] > patch.shape[0] or template.shape[1] > patch.shape[1]:
                    continue
                scores = cv2.matchTemplate(patch, template, cv2.TM_CCORR_NORMED)
                _, score, _, point = cv2.minMaxLoc(scores)
                box = (left + point[0] + 4, top + point[1] + 4,
                       left + point[0] + 4 + w, top + point[1] + 4 + h)
                if score >= .4 and clickable_ring(box) and (best is None or score > best['geometry']):
                    best = dict(glyph=(int(x), int(y), int(width), int(height)), box=box, geometry=score)
        if best is not None:
            candidates.append(best)
    return sorted(candidates, key=lambda item: -item['geometry'])[:8]


def scene_ex_rings(field, reader):
    """只接受完整 EX 字样且识别分数至少 0.95 的几何候选；不把普通文字或孤立亮圈当敌人。

    返回统一客户区检测对象，实际输入仍须两帧稳定及备战身份校验。模板与 OCR 均为通用视觉证据。
    """
    candidates = ex_ring_candidates(field)
    if not candidates:
        return []
    crops = []
    for item in candidates:
        x, y, width, height = item['glyph']
        crop = field[max(0, y - 2):y + height + 2, max(0, x - 2):x + width + 2]
        crops.append(cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
    recognized = reader.paddleocr.recognize(crops)
    result = []
    for item, text in zip(candidates, recognized):
        score = float(text.get('rec_score', 0.))
        if str(text.get('rec_text', '')).strip().upper() == 'EX' and np.isfinite(score) and score >= .95:
            result.append(Detection('scene_enemy_ring', score, tuple(float(value) for value in item['box'])))
    return result


def minimap_enemy_hints(panel):
    """补记小地图上未分类的粉红敌人符号，只作未清提示，不授予任何点击或战斗许可。

    保守排除固定控件及资源图例；有符号但模型分类失败时，清空确认仍须阻止。
    """
    hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (135, 45, 80), (179, 255, 255))
    mask[:29, :42] = mask[:29, -34:] = mask[-31:, -76:] = 0
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    return [dict(kind='minimap_enemy_unclassified', position=point.tolist())
            for box, point in zip(stats[1:], centers[1:])
            if 3 <= box[2] <= 40 and 3 <= box[3] <= 40 and box[4] >= 6]
