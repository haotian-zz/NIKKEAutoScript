"""主线地图右下角的章节号识别，输入为 1776×999 的 BGR 客户区截图。"""

import re

import cv2
import numpy as np


CLIENT_SIZE = (1776, 999)
CHAPTER_AREA = (1608, 917, 1663, 953)
DIFFICULTY_AREA = (1613, 898, 1670, 912)


def chapter_identity(image, reader):
    """固定完整字段优先直接识别，低分或未知版式走原OCR流水线；门槛不变。

    已知单行不需要重复文字检测，完成浮层下变暗的NORMAL仍可直接读字；
    章号至少0.95、难度至少0.9，完整解析后才返回，不从期望章节补字或改数字。
    """
    if image is None or image.shape != (CLIENT_SIZE[1], CLIENT_SIZE[0], 3):
        raise ValueError('章节识别需要 1776×999 的三通道客户区截图。')
    current = fixed_chapter_identity(image, reader)
    if current is not None:
        return current
    values = []
    for area, scale, threshold in ((CHAPTER_AREA, 3, .95), (DIFFICULTY_AREA, 4, .9)):
        x1, y1, x2, y2 = area
        crop = cv2.resize(image[y1:y2, x1:x2], None, fx=scale, fy=scale)
        values.append(reader.ocr(crop, direct_ocr=True, threshold=threshold)['text'].strip())
    number, difficulty = values
    if not re.fullmatch(r'[0-9]{1,2}', number) or int(number) < 1:
        return None
    if difficulty.upper() not in ('NORMAL', 'HARD'):
        return None
    return int(number), difficulty.lower()


def fixed_chapter_identity(image, reader):
    """复用同一识别模型一次读取已裁好的章号与难度两行，不调用文字检测器。

    只接受两个完整字段及有限置信度；未知、额外文字、残缺数字或低分均返回None，
    由调用方选择保守回退。输入须已由chapter_identity核对正式客户区尺寸。
    """
    crops = [cv2.resize(image[top:bottom, left:right], None, fx=scale, fy=scale)
             for (left, top, right, bottom), scale in ((CHAPTER_AREA, 3), (DIFFICULTY_AREA, 4))]
    found = reader.paddleocr.recognize(crops)
    if len(found) != 2:
        return None
    texts = [str(row.get('rec_text', '')).strip() for row in found]
    scores = [float(row.get('rec_score', 0.)) for row in found]
    if (not np.isfinite(scores).all() or scores[0] < .95 or scores[1] < .9
            or re.fullmatch(r'[0-9]{1,2}', texts[0]) is None or int(texts[0]) < 1
            or texts[1].upper() not in ('NORMAL', 'HARD')):
        return None
    return int(texts[0]), texts[1].lower()


def chapter_number(image, model):
    """只读章节号的轻量版本，直接吃 PaddleOCR 模型而不经过项目的 Ocr 封装。

    供标定、离线批处理等没有任务上下文的场合使用，因此不识别难度；放大三倍再识别是因为
    原始数字只有约 36px 高。识别分数低于 0.95、不是一到两位数字或解析出 0 时一律返回 None，
    宁可让调用方重试也不要把误识别的章节号传下去。
    """
    if image is None or image.shape != (CLIENT_SIZE[1], CLIENT_SIZE[0], 3):
        raise ValueError('章节识别需要 1776×999 的三通道客户区截图。')
    x1, y1, x2, y2 = CHAPTER_AREA
    crop = cv2.resize(image[y1:y2, x1:x2], None, fx=3, fy=3)
    result = next(iter(model.predict(crop)), None)
    if result is None:
        return None
    text = result.get('rec_text', '').strip()
    score = result.get('rec_score', 0)
    if not re.fullmatch(r'[0-9]{1,2}', text) or not score >= 0.95:
        return None
    number = int(text)
    return number if number > 0 else None
