"""竖屏章节列表（CHAPTER LIST）里切到困难难度：识别列表、困难页签和已解锁的困难章节卡片。

主线地图右下角的 NORMAL/HARD 标签点开后是章节列表，底部「剧情 / 普通 / 困难」三个页签，
困难页按之字形排列章节卡片：已解锁的卡片上下边框是白线（灰度约 235），未解锁的边框发灰（约 200）
且中间是锁图标。输入均为 720×1280 BGR 客户区截图。
实录 2026-10-23：困难页自动滚到当前进度附近，第 3 章 COMPLETE、第 4 章 0/18，第 5 章起上锁。
"""

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
