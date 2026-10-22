"""章节主线完成浮层识别；只返回界面证据，不将主线完成当作EX及遗失物全清。"""

from functools import lru_cache
from pathlib import Path
import re

import cv2
import numpy as np

MISSION_BAR = (716, 375, 1060, 412)
MISSION_LINE = (765, 378, 1025, 412)
CURRENT_CHAPTER = (718, 249, 806, 282)
NEXT_CONTROL = (470, 560, 1305, 750)
UNLOCK_LINE = (839, 393, 948, 433)
UNLOCK_TITLE = (740, 550, 1040, 610)
UNLOCK_TITLES = {'开放探索', '開放探索', '解锁快速模拟立即完成', '解鎖快速模擬立即完成'}
UNLOCK_PROMPT = (815, 613, 965, 656)
CHAPTER_LIST_LINE = (658, 105, 766, 135)
CHAPTER_LIST_CLOSE = (1075, 95, 1126, 145)


@lru_cache(maxsize=3)
def caption_template(name):
    """缓存只包含固定完成或解锁字形的二值UI资源，不含章节号、地面或人物。"""
    path = Path(__file__).parent/'assets'/name
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(path)
    return template


def mission_caption_score(field):
    """在固定完成白栏内匹配完整字形，不让细斜体和装饰箭头决定章节OCR置信度。

    正文编号仍独立OCR；二值阈值由当前白栏Otsu获得，只匹配UI局部，不根据地图主色判断。
    返回归一相关分数，不是模型概率；后续仍须章节关系及两帧实际控件复核。
    """
    left, top, right, bottom = MISSION_LINE
    gray = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return float(cv2.minMaxLoc(cv2.matchTemplate(
        mask, caption_template('chapter_mission_clear_mask.png'), cv2.TM_CCOEFF_NORMED))[1])


def overlay_anchor(field):
    """廉价识别阻塞地图的章节完成/开放探索字形；不读文字、不推测地面可操作性。

    返回kind和字形相关分数，相关分数不是概率；只有完整核验函数可提供点击许可。
    普通地图仅处理两个固定UI小区域，供每次地图输入前排除被浮层遮挡的画面。
    """
    if field.shape != (999, 1776, 3) or field.dtype != np.uint8:
        raise ValueError('章节浮层识别要求1776×999 uint8客户区。')
    left, top, right, bottom = MISSION_BAR
    hsv = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    if np.mean((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 180)) >= .6:
        score = mission_caption_score(field)
        if np.isfinite(score) and score >= .9:
            return dict(kind='chapter_completion', anchor_score=score)
    left, top, right, bottom = UNLOCK_LINE
    hsv = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (179, 90, 255))
    score = float(cv2.minMaxLoc(cv2.matchTemplate(
        mask, caption_template('chapter_unlock_mask.png'), cv2.TM_CCOEFF_NORMED))[1])
    if np.isfinite(score) and score >= .9:
        return dict(kind='chapter_unlock', anchor_score=score)
    left, top, right, bottom = CHAPTER_LIST_LINE
    gray = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 100)
    score = float(cv2.minMaxLoc(cv2.matchTemplate(
        mask, caption_template('chapter_list_mask.png'), cv2.TM_CCOEFF_NORMED))[1])
    if np.isfinite(score) and score >= .9:
        return dict(kind='chapter_list', anchor_score=score)
    return None


def chapter_list_close(field):
    """定位CHAPTER LIST右上角完整X控件，要求唯一连通块及四角对角笔画，不提供固定落点。

    只在完整列表标题锚点已确认后调用；尺寸14～32px、黑色笔画须至少95%贴合两条对角线，
    四象限均有笔画，以排除边框、加号和单斜线。返回实际客户区包围框或None。
    """
    left, top, right, bottom = CHAPTER_LIST_CLOSE
    gray = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 0, 100)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for index, (x, y, width, height, area) in enumerate(stats[1:count], 1):
        if not (14 <= width <= 32 and 14 <= height <= 32 and .85 <= width/height <= 1.15
                and .15 <= area/(width*height) <= .45):
            continue
        ys, xs = np.nonzero(labels[y:y+height, x:x+width] == index)
        xx, yy = xs/(width-1), ys/(height-1)
        if np.mean(np.minimum(abs(xx-yy), abs(xx+yy-1)) <= .09) < .95:
            continue
        if any(np.count_nonzero(((xx < .4) if a == 0 else (xx > .6))
                                & ((yy < .4) if b == 0 else (yy > .6))) < 5
               for a in (0, 1) for b in (0, 1)):
            continue
        boxes.append([int(left+x), int(top+y), int(left+x+width), int(top+y+height)])
    return boxes[0] if len(boxes) == 1 else None


def chapter_number(text, caption=False):
    """解析明确CHAPTER前缀后的完整章节号，拒绝截断数字和OCR尾随字母。

    当前章徽标须只有CHAPTER及编号；下一章控件允许冒号分隔或中文名称，不从正文猜数字。
    不支持的标题排版保持未确认，不能以期望章节补齐。
    """
    match = re.fullmatch(r'\s*CHAPTER\s*([1-9]\d?)(.*)', text, re.I)
    if match is None:
        return None
    tail = match.group(2).strip()
    if tail and (not caption or not re.match(r'[:：|·\-\u4e00-\u9fff]', tail)):
        return None
    return int(match.group(1))


def trim_line(crop, ink):
    """按同一行字形高度收紧识别范围，去除宽间距标题旁的小箭头与控件边框。

    只调整裁图外接框，不删除文字像素或补写字符。取完整高字形并保留2px边距；
    无可靠字形时返回原图，避免将短片段伪装成完整章节标题。
    """
    height, width = crop.shape[:2]
    if ink == 'dark':
        mask = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255,
                             cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 190), (179, 90, 255))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = [item for item in stats[1:count] if 6 <= item[3] <= .9*height
              and item[2] <= .9*width and item[4] >= 4 and item[0] > 0 and item[1] > 0
              and item[0]+item[2] < width and item[1]+item[3] < height]
    if not glyphs:
        return crop
    minimum = .65*max(item[3] for item in glyphs)
    glyphs = [item for item in glyphs if item[3] >= minimum]
    if len(glyphs) < 5:
        return crop
    left = max(0, min(item[0] for item in glyphs)-2)
    top = max(0, min(item[1] for item in glyphs)-2)
    right = min(width, max(item[0]+item[2] for item in glyphs)+2)
    bottom = min(height, max(item[1]+item[3] for item in glyphs)+2)
    return crop[top:bottom, left:right]


def read_line(field, roi, reader, ink=None, scale=2):
    """对固定浮层中的完整一行直接识别；可去除装饰空白，非有限分数不会获得输入许可。"""
    left, top, right, bottom = roi
    crop = field[top:bottom, left:right]
    if ink is not None:
        crop = trim_line(crop, ink)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = reader.paddleocr.recognize([crop])
    if len(results) != 1:
        return '', 0.
    return results[0].get('rec_text', ''), float(results[0].get('rec_score', 0.))


def next_control_boxes(field):
    """只在浮层下一章控件区域寻找完整蓝色矩形，返回实际客户区框，不提供固定点击兜底。"""
    left, top, right, bottom = NEXT_CONTROL
    hsv = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 70, 50), (120, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for x, y, width, height, area in stats[1:count]:
        if 160 <= width <= 520 and 28 <= height <= 80 and area/(width*height) >= .35:
            if x > 1 and y > 1 and x+width < right-left-1 and y+height < bottom-top-1:
                boxes.append([int(left+x), int(top+y), int(left+x+width), int(top+y+height)])
    return boxes


def control_title_prefix(field, box):
    """以控件内成对标点列分开CHAPTER编号与章名，保留完整数字，不按期望章号裁字。

    数字与字母的高字形全部取到分隔标点前；没有唯一分隔证据时返回None。
    仅调整OCR范围，解决章名和边框降低编号分数的问题，不替换I/1或补写下一章。
    """
    left, top, right, bottom = box
    crop = field[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 190), (179, 90, 255))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    columns = {}
    for x, y, width, height, area in stats[1:count]:
        # 实录 2026-10-21（21→22“侵略”）：分隔点渲染偏小，六个点中四个只有 1 像素；
        # 成对列、间距、跨度及唯一性要求不变。
        if 1 <= width <= 4 and 1 <= height <= 4 and 1 <= area <= 20:
            columns.setdefault(int(x), []).append((int(y), int(width)))
    columns = [(x, min(y for y, _ in dots), max(y for y, _ in dots), max(w for _, w in dots))
               for x, dots in sorted(columns.items()) if len(dots) >= 2
               and max(y for y, _ in dots)-min(y for y, _ in dots) >= 4]
    pairs = [(a, b) for index, a in enumerate(columns) for b in columns[index+1:]
             if 3 <= b[0]-a[0] <= 12 and abs(a[1]-b[1]) <= 2 and abs(a[2]-b[2]) <= 2]
    if len(pairs) != 1:
        return None
    first, last = pairs[0]
    glyphs = [item for item in stats[1:count] if item[0] > 0 and item[1] > 0
              and item[0]+item[2] < first[0] and item[3] >= 8 and item[4] >= 5]
    if len(glyphs) < 5:
        return None
    height = float(np.median([item[3] for item in glyphs]))
    # 发光控件外框会扩缩，白色竖框可能进入蓝色包围框；按正文主高度排除它，保留完整编号。
    glyphs = [item for item in glyphs if .75*height <= item[3] <= 1.25*height]
    if len(glyphs) < 5:
        return None
    x1 = max(0, min(item[0] for item in glyphs)-round(.5*height))
    y1 = max(0, min(item[1] for item in glyphs)-round(.15*height))
    x2 = min(crop.shape[1], last[0]+last[3])
    y2 = min(crop.shape[0], max(item[1]+item[3] for item in glyphs)+round(.3*height))
    return [int(left+x1), int(top+y1), int(left+x2), int(top+y2)]


def completion_overlay(field, reader):
    """结合完整MISSION CLEAR字形、当前章徽标及实际下一章蓝控件核验阻塞浮层。

    正式1776×999 BGR客户区；白色标题栏先做廉价筛选，普通地图不持续调用OCR。
    识别到完成字样但编号/控件不可靠时返回accepted=False，调用方应保留阻塞；
    字形相关及两行编号OCR均≥0.9、下一章恰为当前章+1，才返回供双帧复核的控件中心。
    本函数不发输入，也不把主线完成认定为可选内容全清。
    """
    if field.shape != (999, 1776, 3) or field.dtype != np.uint8:
        raise ValueError('章节完成浮层识别要求1776×999 uint8客户区。')
    left, top, right, bottom = MISSION_BAR
    hsv = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    if np.mean((hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 180)) < .6:
        return None
    mission_score = mission_caption_score(field)
    if not np.isfinite(mission_score) or mission_score < .9:
        return None
    result = dict(kind='chapter_completion', accepted=False,
                  mission_anchor='binary_glyph', mission_score=mission_score, current=None, following=None, point=None)
    boxes = next_control_boxes(field)
    if len(boxes) != 1:
        return dict(result, reason='next_control_ambiguous', controls=boxes)
    prefix = control_title_prefix(field, boxes[0])
    if prefix is None:
        return dict(result, reason='next_title_separator_unconfirmed', box=boxes[0])
    current_text, current_score = read_line(field, CURRENT_CHAPTER, reader, ink='dark')
    next_text, next_score = read_line(field, prefix, reader, scale=3)
    current, following = chapter_number(current_text), chapter_number(next_text)
    accepted = (np.isfinite([current_score, next_score]).all() and min(current_score, next_score) >= .9
                and current is not None and following == current+1)
    result.update(accepted=bool(accepted), current=current, following=following, current_text=current_text,
                  next_text=next_text, current_score=current_score, next_score=next_score, box=boxes[0])
    if accepted:
        result['point'] = ((np.asarray(boxes[0][:2])+boxes[0][2:])/2).tolist()
    else:
        result['reason'] = 'chapter_titles_unconfirmed'
    return result


def chapter_overlay(field, reader):
    """完整核验已知章节浮层；解锁须独立识别完整正文及继续文字，不能凭UNLOCK直接点击。

    只接受已实测的开放探索及快速模拟立即完成两类正文；未知语言/正文保持阻塞且point=None。
    不修改章节清理记录，不把解锁、浮层消失或按钮响应视为关卡胜利或全清。
    """
    anchor = overlay_anchor(field)
    if anchor is None:
        return None
    if anchor['kind'] == 'chapter_completion':
        return completion_overlay(field, reader)
    if anchor['kind'] == 'chapter_list':
        box = chapter_list_close(field)
        return dict(anchor, accepted=box is not None, box=box,
                    point=None if box is None else [(box[0]+box[2])/2, (box[1]+box[3])/2])
    title, title_score = read_line(field, UNLOCK_TITLE, reader, scale=3)
    prompt, prompt_score = read_line(field, UNLOCK_PROMPT, reader, scale=3)
    accepted = (np.isfinite([title_score, prompt_score]).all() and min(title_score, prompt_score) >= .9
                and title.strip() in UNLOCK_TITLES
                and re.fullmatch(r'\s*[点點][击擊][继繼][续續][。.]?\s*', prompt) is not None)
    result = dict(anchor, accepted=bool(accepted), title=title, title_score=title_score,
                  prompt=prompt, prompt_score=prompt_score, point=None)
    if accepted:
        left, top, right, bottom = UNLOCK_PROMPT
        hsv = cv2.cvtColor(field[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        ys, xs = np.nonzero(cv2.inRange(hsv, (0, 0, 190), (179, 90, 255)))
        if len(xs) >= 20:
            result['box'] = [int(left+xs.min()), int(top+ys.min()), int(left+xs.max()+1), int(top+ys.max()+1)]
            result['point'] = [(result['box'][0]+result['box'][2])/2, (result['box'][1]+result['box'][3])/2]
        else:
            result.update(accepted=False, reason='continue_prompt_geometry_unconfirmed')
    else:
        result['reason'] = 'exploration_text_unconfirmed'
    return result
