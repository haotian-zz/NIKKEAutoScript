"""即时通讯（blabla）「任务」页：读完未读对话，解锁章节支线委托。

对话末尾会发放「【第N章】清理/搜索……所提到的地点」委托（每个 50 钻），只有读完对话才会出现在
章节地图上。读对话不花任何资源，回复选项只有一个分支时直接点，多个时点最下面那个。

画面识别全部基于 720×1280 竖屏客户区的颜色布局，不依赖文字 OCR：
- 列表与对话页都是橙色顶栏；列表的橙色延伸到页签行，面板左缘 (44~56, 280~300) 为橙色。
- 对话页顶栏左侧有白色返回箭头，面板左缘是灰底（已发送的橙色气泡靠右）。
- 回复选项是橙色外框包住白色内框，左缘约 x=277；已发送的气泡是橙底白字，白色占比低。
- 未读对话行右侧有红色 N 或红点。
- 对话进行中，底部总有回复选项或「… ▼」等待点击的气泡；两者都没有就是读到 END 了。
- 有的对话停在玩家刚发出的一句上，要等对方过一段时间才回复（行右侧是红点而不是 N），点开也没什么可读；
  按行标题截图记下来跳过，不能停下整轮。读完的对话留在原位只是去掉标记，后面的未读行要往下滚动才看得到。
  不用画面差分：面板底纹有持续噪点（静止帧也有约 1500 像素变化），委托卡片和点击特效也一直在动。
实录 2026-10-23：34 个任务对话未读，章节 18~24 的黄色支线标记全部为 0。
"""
import time
from pathlib import Path

import cv2
import numpy as np

from module.execution.context import report
from module.logger import logger
from module.ui.page import page_main
from module.ui.ui import UI

ASSETS = Path(__file__).resolve().parents[2] / 'assets/zh-CN/messenger'
ICON = cv2.imread(str(ASSETS / 'MESSENGER_ICON.png'))
# 「… ▼」气泡；三个点在闪，实测 0.77~1.0，没有气泡时约 0.44
TYPING = cv2.imread(str(ASSETS / 'CHAT_TYPING.png'))
TYPING_REGION = (140, 270, 300, 1165)
TYPING_THRESHOLD = .7
# 大厅左侧图标列；活动多时图标顺序会变，所以按模板在整列里找
ICON_REGION = (0, 100, 120, 720)
ICON_THRESHOLD = .85
BACK = (80, 205)
CLOSE = (645, 140)
CHAT_TAP = (360, 1120)
END_ROUNDS = 3
MAX_STEPS = 400
MAX_CHATS = 60
LIST_SWIPE = ((360, 1000), (360, 520))
LIST_SCROLL_ROUNDS = 12


def orange(image):
    b, g, r = cv2.split(image)
    return ((b < 90) & (g > 120) & (g < 205) & (r > 225)).astype(np.uint8)


def red(image):
    b, g, r = cv2.split(image)
    return ((b < 90) & (g < 100) & (r > 200)).astype(np.uint8)


def view(image):
    """'list'：任务列表；'chat'：对话内；'other'：其它画面。image 为 BGR。"""
    if orange(image[120:160, 44:56]).mean() < .8 or orange(image[180:220, 100:620]).mean() < .6:
        return 'other'
    if orange(image[280:300, 44:56]).mean() > .8:
        return 'list'
    if (image[185:225, 68:95].min(axis=2) > 235).mean() > .05:
        return 'chat'
    return 'other'


def reply_options(image):
    """回复选项的点击点 [(x, y)]，自上而下。"""
    mask = orange(image)
    mask[:270] = 0
    mask[1165:] = 0
    found = []
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < 300 or h < 50 or x > 310:
            continue
        white = (image[y:y + h, x:x + w].min(axis=2) > 235).mean()
        if white > .35:
            found.append((y, x + w // 2 + 40, y + h // 2))
    return [(x, y) for _, x, y in sorted(found)]


def unread_rows(image):
    """列表里右侧带红色 N/红点的行，返回行中心 y（只取面板内可安全点击的行）。"""
    mask = red(image[370:1160, 610:665])
    rows = np.where(mask.sum(axis=1) > 3)[0]
    ys, last = [], -100
    for r in rows:
        if r - last > 40 and 370 + r + 35 < 1150:
            ys.append(370 + r + 35)
        last = r
    return ys


def typing(image):
    """对话还没读完：画面里有「… ▼」等待点击的气泡。"""
    x1, y1, x2, y2 = TYPING_REGION
    result = cv2.matchTemplate(image[y1:y2, x1:x2], TYPING, cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(result)[1] >= TYPING_THRESHOLD


def row_title(image, y):
    """列表行的标题区截图，用来认出已经判定为「等待回复」的对话。"""
    return image[y - 32:y - 6, 200:560].astype(np.int16)


def same_row(a, b):
    return a.shape == b.shape and float(np.abs(a - b).mean()) < 8


def find_icon(image):
    """大厅即时通讯图标中心，找不到返回 None。"""
    x1, y1, x2, y2 = ICON_REGION
    result = cv2.matchTemplate(image[y1:y2, x1:x2], ICON, cv2.TM_CCOEFF_NORMED)
    _, value, _, loc = cv2.minMaxLoc(result)
    if value < ICON_THRESHOLD:
        return None
    h, w = ICON.shape[:2]
    return x1 + loc[0] + w // 2, y1 + loc[1] + h // 2


class Messenger(UI):
    def bgr(self):
        self.device.screenshot()
        return cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)

    def tap(self, point, wait):
        self.device.click_xy(int(point[0]), int(point[1]))
        time.sleep(wait)

    def wait_view(self, name, seconds=15):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            image = self.bgr()
            if view(image) == name:
                return image
            time.sleep(.5)
        return None

    def open(self):
        self.ui_ensure(page_main)
        icon = find_icon(self.bgr())
        if icon is None:
            return False
        self.tap(icon, 2)
        return self.wait_view('list') is not None

    def read_chat(self):
        """读完当前打开的对话，返回 (选项次数, 推进次数)。"""
        picks = taps = idle = 0
        for _ in range(MAX_STEPS):
            image = self.bgr()
            if view(image) != 'chat':
                logger.warning(f'Messenger: unexpected view {view(image)} inside chat')
                break
            options = reply_options(image)
            if options:
                self.tap(options[-1], 1.2)
                picks, idle = picks + 1, 0
            elif typing(image):
                self.tap(CHAT_TAP, 1.0)
                taps, idle = taps + 1, 0
            else:
                # 选项刚点下、下一句还没出来时也会短暂两者都没有，连续几轮都没有才算读完
                idle += 1
                if idle >= END_ROUNDS:
                    break
                time.sleep(1.0)
        return picks, picks + taps

    def list_image(self):
        image = self.bgr()
        if view(image) == 'chat':
            self.tap(BACK, 2)
            image = self.bgr()
        if view(image) != 'list':
            raise RuntimeError(f'Messenger list not visible: {view(image)}')
        return image

    def scroll_list(self, image, down):
        """滚动对话列表；返回新画面，滚不动（到底/到顶）时返回 None。"""
        start, end = LIST_SWIPE if down else LIST_SWIPE[::-1]
        self.device.swipe(start, end, method='swipe', handle_control_check=False)
        time.sleep(1.2)
        after = self.bgr()
        moved = cv2.absdiff(after[380:1150, 60:660], image[380:1150, 60:660]).mean() > 3
        return after if moved else None

    def next_unread(self, waiting):
        """从列表顶部往下找第一行未读且不在等待名单里的对话，返回 (画面, 行 y)。"""
        image = self.list_image()
        for _ in range(LIST_SCROLL_ROUNDS):
            image = self.scroll_list(image, down=False)
            if image is None:
                break
        image = self.bgr()
        for _ in range(LIST_SCROLL_ROUNDS):
            for y in unread_rows(image):
                title = row_title(image, y)
                if not any(same_row(title, w) for w in waiting):
                    return image, y
            image = self.scroll_list(image, down=True)
            if image is None:
                break
        return None, None

    def read_all(self):
        """读完所有能读的未读对话，返回 (读完数, 等待回复数)。"""
        done, waiting = 0, []
        for _ in range(MAX_CHATS):
            image, y = self.next_unread(waiting)
            if image is None:
                break
            title = row_title(image, y)
            self.tap((360, y), 2.5)
            picks, progress = self.read_chat()
            logger.info(f'Messenger: chat picks={picks} progress={progress}')
            self.tap(BACK, 2)
            if progress:
                done += 1
            else:
                waiting.append(title)
        return done, len(waiting)

    def run(self):
        logger.hr('Messenger', 1)
        try:
            if not self.open():
                report('skipped', '大厅没有找到即时通讯图标或列表未打开')
            else:
                done, waiting = self.read_all()
                report('success', f'即时通讯已读完 {done} 个任务对话（解锁章节支线委托），{waiting} 个在等对方回复')
                self.tap(CLOSE, 2)
        except Exception as e:
            report('failed', '即时通讯对话异常结束', type(e).__name__, retryable=True)
            logger.error(e)
        self.ui_ensure(page_main)
        self.config.task_delay(schedule=True)
