"""竖屏章节列表（CHAPTER LIST）里切到困难难度：识别列表、困难页签、已解锁的困难章节卡片及其进度计数。

主线地图右下角的 NORMAL/HARD 标签点开后是章节列表，底部「剧情 / 普通 / 困难」三个页签，
困难页按之字形排列章节卡片：已解锁的卡片上下边框是白线（灰度约 235），未解锁的边框发灰（约 200）
且中间是锁图标。输入均为 720×1280 BGR 客户区截图。
实录 2026-10-23：困难页自动滚到当前进度附近，第 3 章 COMPLETE、第 4 章 0/18，第 5 章起上锁。
"""

import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'
LIST_TAB = cv2.imread(str(ASSETS / 'chapter_list_tab.png'))
if LIST_TAB is None:
    raise FileNotFoundError(ASSETS / 'chapter_list_tab.png')
# 只比对「剧情」页签左侧 3/4：有新剧情时右上角会叠红色 N 角标。
# 实录 2026-10-24：带角标时整张模板只有 0.71，去掉右侧后 0.997，其它画面最高 0.59。
LIST_TAB = LIST_TAB[:, :LIST_TAB.shape[1] * 3 // 4]
LIST_TAB_REGION = (0, 1050, 360, 1200)
LIST_TAB_THRESHOLD = .9
# 地图右下角难度标签（竖屏）、困难页签、关闭按钮
DIFFICULTY_TAG = (540, 1158)
HARD_TAB = (583, 1123)
HARD_TAB_AREA = (510, 1103, 660, 1145)
LIST_CLOSE = (630, 153)
# 卡片在左右两列交替排列
CARD_COLUMNS = ((68, 258), (462, 652))
LIST_AREA = (180, 1090)
BORDER_GRAY = 220
CARD_HEIGHT = (30, 50)
# 困难页截图样本，供进度计数复查，最多保留 SAMPLE_LIMIT 张。
SAMPLE_FOLDER = Path('log/campaign/chapter_list')
SAMPLE_LIMIT = 20
# 卡片标题框下方的进度计数（相对标题框中心，720×1280）。实录 2026-10-25 困难页：
# 第 7 章 收集品 1/5、蓝色关卡 34/34、红色关卡 2/2；第 8 章 3/5、27/27、2/2；第 9 章 0/5、12/27、0/2。
# 蓝色、红色两项都满即本章关卡全部通关（收集品不计）。
COUNTER_AREAS = dict(stage=(-48, 58, 4, 86), main=(26, 58, 70, 86))
COUNTER_ROWS = (-95, 28, 95, 88)
# 计数是空心描边字体，OCR 对 2、7 常漏读或把斜杠读成 1（27/27 → 27127），不能单独判定是否满。
# n==N 时斜杠两侧的字形逐像素相同：取文字最左、最右等宽两段做归一化互相关，满的样本 0.99，
# 12/27 只有 0.61。中间只留斜杠的宽度。
COUNTER_LEVEL = 120
COUNTER_SATURATION = 70
COUNTER_SAME = .95
SLASH_WIDTH = (5, 16)


def list_visible(image):
    x1, y1, x2, y2 = LIST_TAB_REGION
    result = cv2.matchTemplate(image[y1:y2, x1:x2], LIST_TAB, cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(result)[1] >= LIST_TAB_THRESHOLD


def hard_tab_active(image):
    """困难页签选中时是暗红底（实测 BGR 约 55,58,156），未选中是白底。"""
    x1, y1, x2, y2 = HARD_TAB_AREA
    b, g, r = image[y1:y2, x1:x2].reshape(-1, 3).mean(0)
    return r > 120 and r - g > 60 and r - b > 60


def unlocked_cards(image):
    """已解锁章节卡片的中心点，按 y 从上到下排序。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    top, bottom = LIST_AREA
    cards = []
    for x1, x2 in CARD_COLUMNS:
        bright = np.median(gray[top:bottom, x1 + 10:x2 - 10], axis=1) >= BORDER_GRAY
        rows = np.nonzero(bright)[0]
        if not len(rows):
            continue
        lines = [int(run.mean()) + top for run in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)]
        for upper, lower in zip(lines, lines[1:]):
            if CARD_HEIGHT[0] <= lower - upper <= CARD_HEIGHT[1]:
                cards.append(((x1 + x2) // 2, (upper + lower) // 2))
    return sorted(cards, key=lambda point: point[1])


def save_sample(image):
    """保存困难页截图（BGR）供进度计数识别取样；超过上限删最旧的。"""
    SAMPLE_FOLDER.mkdir(parents=True, exist_ok=True)
    path = SAMPLE_FOLDER / f'{datetime.now():%Y%m%d_%H%M%S}_hard.png'
    cv2.imwrite(str(path), image)
    for old in sorted(SAMPLE_FOLDER.glob('*_hard.png'))[:-SAMPLE_LIMIT]:
        old.unlink(missing_ok=True)
    return path


def counter_symmetry(image, area):
    """计数文字斜杠两侧字形的最高相似度（-1～1）；读不到文字返回 None。"""
    x1, y1, x2, y2 = area
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 只取灰白描边，排除左侧蓝色／红色感叹号图标。
    text = (hsv[..., 2] >= COUNTER_LEVEL) & (hsv[..., 1] <= COUNTER_SATURATION)
    columns = np.nonzero(text.sum(0) >= 2)[0]
    if len(columns) < 2*4+SLASH_WIDTH[0]:
        return None
    left, right = columns[0], columns[-1]+1
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    best = None
    for width in range(4, (right-left-SLASH_WIDTH[0])//2+1):
        if not SLASH_WIDTH[0] <= right-left-2*width <= SLASH_WIDTH[1]:
            continue
        a, b = gray[:, left:left+width], gray[:, right-width:right]
        a, b = a-a.mean(), b-b.mean()
        score = float((a*b).sum()/max(np.sqrt((a*a).sum()*(b*b).sum()), 1e-6))
        best = score if best is None else max(best, score)
    return best


def parse_counter(text):
    """OCR 文字里的 n/N；读不出返回 None。"""
    match = re.fullmatch(r'\D*(\d{1,3})/(\d{1,3})\D*', text.replace(' ', ''))
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def card_progress(image, card, ocr=None):
    """卡片进度：dict(complete=True/False/None, stage=..., main=..., text=...)。

    complete 为 True 需要卡片上有 COMPLETE 字样，或蓝、红两项计数斜杠两侧字形都相同；
    OCR 只用于日志里的剩余关数，读到明确 n<N 时否决「已满」。读不到计数时 complete 为 None。
    ocr(area) 返回该区域文字，可为 None（只做字形比较）。
    """
    cx, cy = card
    def absolute(offsets):
        dx1, dy1, dx2, dy2 = offsets
        return cx+dx1, cy+dy1, cx+dx2, cy+dy2
    rows = ocr(absolute(COUNTER_ROWS)) if ocr is not None else ''
    if 'COMPLETE' in rows.upper().replace(' ', ''):
        return dict(complete=True, stage=None, main=None, text=rows)
    result = dict(text=rows)
    full = []
    for name, offsets in COUNTER_AREAS.items():
        area = absolute(offsets)
        same = counter_symmetry(image, area)
        read = parse_counter(ocr(area)) if ocr is not None else None
        result[name] = dict(same=None if same is None else round(same, 3), read=read)
        if same is None:
            full.append(None)
        elif read is not None and read[0] < read[1]:
            full.append(False)
        else:
            full.append(same >= COUNTER_SAME)
    result['complete'] = None if None in full else all(full)
    return result
