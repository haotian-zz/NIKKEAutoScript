"""即时通讯任务对话里的章节支线委托：点「直接前往」跳到章节地图，逐个点橙色「!」菱形打 SUB 关卡。

每个 SUB 关卡首通直接给钻石（实录 2026-10-23：SUB 18-6/18-7/18-8 各 50 钻），委托本身另有奖励。
委托只有读完对话才会出现，所以通常排在 Messenger 任务之后。

画面识别全部在 720×1280 竖屏客户区：
- 对话顶部的委托卡片：未完成时是白底「直接前往」按钮（橙色箭头），完成后变成橙底「可完成」，点了领奖励。
- 点按钮后游戏直接进入该章普通地图，镜头在小队身上；右下角章节组件下方的「CURRENT AREA」字样作为地图锚点。
- 支线目标头顶是橙框白底「!」菱形；离屏目标在屏幕边缘显示同样的菱形（带小箭头）。
  点场内菱形会由游戏原生寻路走到敌人并弹出备战；点边缘菱形会把镜头移过去，下一轮再点场内菱形。
- 备战弹窗标题为「SUB 章-关 STAGE」才进入战斗；主线关卡、EX 关卡等其它标题一律关闭弹窗并停止本章，
  不替主线推图做决定。
"""
import re
import time
from pathlib import Path

import cv2
import numpy as np

from module.execution.context import report
from module.logger import logger
from module.messenger.messenger import BACK, CLOSE, LIST_SCROLL_ROUNDS, Messenger, orange, typing, view
from module.ocr.ocr import Ocr
from module.simulation_room.assets import AUTO_BURST, AUTO_SHOOT, END_FIGHTING, FIGHT
from module.tribe_tower.assets import OPERATION_FAILED
from module.ui.assets import FIGHT_CLOSE
from module.ui.page import page_main

ASSETS = Path(__file__).resolve().parents[2] / 'assets/zh-CN/side_quest'
# 对话顶部委托卡片的「直接前往」按钮（橙色箭头＋文字）
QUEST_GO = cv2.imread(str(ASSETS / 'QUEST_GO.png'))
QUEST_GO_REGION = (300, 270, 700, 700)
QUEST_GO_THRESHOLD = .85
# 委托完成后同一位置变成橙底「可完成」按钮，点了领奖励
QUEST_DONE = cv2.imread(str(ASSETS / 'QUEST_DONE.png'))
REWARD_CLOSE = (360, 1000)
# 对话顶部委托卡片上的任务文字（如「【第24章】清理……地点」），只用于日志和复盘
CARD_TEXT = (160, 400, 455, 462)
# 委托没打出 SUB 或中途停止时的画面存这里
EVIDENCE = Path('log/side_quest')
# 「寻找某人」类支线：点菱形后弹的不是备战而是带蓝色「对话」按钮的 SUB QUEST 弹窗，点了播一段剧情就算完成
QUEST_TALK = cv2.imread(str(ASSETS / 'QUEST_TALK.png'))
QUEST_TALK_REGION = (200, 1100, 520, 1260)
TALK_CLOSE = (703, 738)
QUEST_TALK_POINT = (360, 1188)
# 小队已经走到目标旁边时「!」浮在小队头上，点它等于点自己；敌人本体在菱形下方约 110px
ENEMY_BELOW_MARKER = 110
# 地图右下角章节组件的「CURRENT AREA」
MAP_CHECK = cv2.imread(str(ASSETS / 'MAP_CHECK.png'))
MAP_CHECK_REGION = (440, 1190, 640, 1260)
MAP_CHECK_THRESHOLD = .85
MAP_HOME = (195, 1208)
# 菱形外框尺寸约 76～91px，被屏幕边缘截掉一半的也要认（实录宽 59px）
MARKER_SIZE = (50, 100)
# 小地图、任务栏、底部按钮和章节组件上不点
UNSAFE = ((0, 0, 720, 70), (0, 90, 300, 380), (390, 90, 720, 220), (0, 1140, 720, 1280))
ENTRY_TITLE = (25, 685, 330, 745)
ENTRY_WAIT = 12.
BATTLE_WAIT = 600.
MAP_WAIT = 90.
MAX_FIGHTS = 30
# 连续这么多次点菱形都没弹备战就放弃本章（约两分钟寻路）
MAX_MISSES = 10
# 画面里没有可点的菱形时依次拖镜头看上、下、右、左（内容跟着手指走），都没有才算本章打完。
# 实录 2026-10-23（18 章）：打完附近三个后剩下两个菱形压在顶部资源栏下，没有边缘提示。
# 实测拖 450px 镜头只移约 230px，所以每次拖满一屏，上下左右各看两屏再回来。
VIEW_UP, VIEW_DOWN = ((360, 300), (360, 1000)), ((360, 1000), (360, 300))
VIEW_RIGHT, VIEW_LEFT = ((620, 640), (100, 640)), ((100, 640), (620, 640))
SWEEP = ((VIEW_UP,) * 2 + (VIEW_DOWN,) * 4 + (VIEW_UP,) * 2
         + (VIEW_RIGHT,) * 2 + (VIEW_LEFT,) * 4)
MAX_CHATS = 60
# 对话列表每行左侧的黄色头像
AVATAR_COLUMN = (90, 170)
LIST_TOP, LIST_BOTTOM = 380, 1160


def find_template(image, template, region, threshold):
    """模板在区域里的中心点，找不到返回 None。"""
    x1, y1, x2, y2 = region
    result = cv2.matchTemplate(image[y1:y2, x1:x2], template, cv2.TM_CCOEFF_NORMED)
    _, value, _, loc = cv2.minMaxLoc(result)
    if value < threshold:
        return None
    h, w = template.shape[:2]
    return x1 + loc[0] + w // 2, y1 + loc[1] + h // 2


def quest_go(image):
    return find_template(image, QUEST_GO, QUEST_GO_REGION, QUEST_GO_THRESHOLD)


def quest_done(image):
    return find_template(image, QUEST_DONE, QUEST_GO_REGION, QUEST_GO_THRESHOLD)


def quest_talk(image):
    return find_template(image, QUEST_TALK, QUEST_TALK_REGION, QUEST_GO_THRESHOLD)


def map_visible(image):
    return find_template(image, MAP_CHECK, MAP_CHECK_REGION, MAP_CHECK_THRESHOLD) is not None


def quest_markers(image):
    """橙框白底「!」菱形的中心 [(x, y)]，按离屏幕中心由近到远排序。

    橙色外框按连通块取外接框，尺寸接近正方形；中心要有白底（排除火焰、小地图图标等橙色物体）。
    落在小地图、顶栏等界面下面的不返回：那里点不到，而且半透明顶栏会把菱形压暗认不出来。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (8, 150, 200), (25, 255, 255))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    low, high = MARKER_SIZE
    found = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if not (low <= w <= high and low <= h <= high) or max(w, h) > 1.5 * min(w, h):
            continue
        cx, cy = x + w // 2, y + h // 2
        inner = image[max(cy - 8, 0):cy + 8, max(cx - 8, 0):cx + 8]
        if inner.size == 0 or (inner.min(axis=2) > 200).mean() < .3:
            continue
        if not any(a <= cx < c and b <= cy < d for a, b, c, d in UNSAFE):
            found.append((cx, cy))
    return sorted(found, key=lambda p: (p[0] - 360) ** 2 + (p[1] - 640) ** 2)


def sub_stage(text):
    """备战标题「SUB 18-8 STAGE」→ (18, 8)；其它标题返回 None。"""
    match = re.search(r'SUB\s*([1-9]\d?)\s*[-－]\s*([1-9]\d?)', text or '', re.I)
    return tuple(map(int, match.groups())) if match else None


def chat_rows(image):
    """对话列表里完整显示的行：按左侧黄色头像的竖向连通段取行中心 y。"""
    x1, x2 = AVATAR_COLUMN
    column = orange(image[LIST_TOP:LIST_BOTTOM, x1:x2]).mean(axis=1) > .3
    rows, start = [], None
    for i, on in enumerate(np.append(column, False)):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= 60:
                rows.append(LIST_TOP + (start + i) // 2)
            start = None
    return rows


def row_title(image, y):
    """列表行的标题截图，用来记住已经看过的对话。"""
    return image[y - 35:y - 5, 200:560].astype(np.int16)


def same_row(a, b):
    return a.shape == b.shape and float(np.abs(a - b).mean()) < 8


def title_key(text):
    """对话标题文字去掉菱形图标、标点和空白，只留文字和数字用于比对。"""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text or '')


def same_chat(a, b):
    """两个列表行是不是同一个对话：两边都读出标题文字就比文字，否则退回截图比对。

    实录 2026-10-25：打完 24 章 SUB 回列表领奖，对话已移到顶部且带红点，标题截图差异超过门限，
    没找回对话，「可完成」的 50 钻一直没领；探针按截图去重也把同一个对话反复打开了四次。
    """
    if a['text'] and b['text']:
        return a['text'] == b['text']
    return same_row(a['image'], b['image'])


class SideQuest(Messenger):
    @property
    def entry_ocr(self):
        return Ocr(ENTRY_TITLE, name='SIDE_QUEST_ENTRY', model_type=self.config.Optimization_OcrModelType)

    def row_identity(self, image, y):
        """列表行标题：OCR 文字为主，读不出时用截图（对话更新后行会移到顶部、带红点，像素会变）。"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ocr = Ocr((200, y - 35, 560, y - 5), name='SIDE_QUEST_ROW', model_type=self.config.Optimization_OcrModelType)
        return dict(text=title_key(ocr.ocr(rgb, threshold=.6)['text']), image=row_title(image, y))

    def card_text(self):
        """当前对话顶部委托卡片的任务文字；读不出返回空串。"""
        self.device.screenshot()
        ocr = Ocr(CARD_TEXT, name='SIDE_QUEST_CARD', model_type=self.config.Optimization_OcrModelType)
        return ocr.ocr(self.device.image, threshold=.6)['text'] or ''

    def save_evidence(self, label):
        """存一张当前画面供复盘；写不了文件不影响流程。"""
        try:
            EVIDENCE.mkdir(parents=True, exist_ok=True)
            path = EVIDENCE / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.png"
            cv2.imwrite(str(path), self.bgr())
            return path
        except (OSError, cv2.error):
            return None

    def wait_map(self, seconds=MAP_WAIT):
        """等到章节地图或备战弹窗：返回 ('map'|'entry', 画面)，超时返回 (None, None)。

        寻路远时备战弹窗会在点菱形十几秒后才弹出，弹窗盖住地图锚点，所以两者一起等。
        实录 2026-10-23（SUB 18-3）：只等地图时弹窗一直开着，脚本干等到超时。
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.popup() is not None:
                return 'entry', None
            # 上一场的结算画面还在（中断后重跑）就点掉
            if self.appear_then_click(END_FIGHTING, offset=30, interval=2):
                time.sleep(1)
                continue
            image = cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)
            if map_visible(image):
                return 'map', image
            time.sleep(1)
        return None, None

    def popup(self):
        """截一张图，返回点菱形后弹出的窗口类型：'fight'（备战）/ 'talk'（对话）/ None。"""
        self.device.screenshot()
        if self.appear(FIGHT, offset=30):
            return 'fight'
        if quest_talk(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)) is not None:
            return 'talk'
        return None

    def engage(self):
        """弹窗已出现：备战就打，对话就点「对话」看完剧情。返回 'win' / 'talk' / 'defeat' / 'skip'。

        实录 2026-10-23：第19章「寻找伏特Jr.」点菱形弹出 SUB QUEST 对话窗，点「对话」后黑屏加载、
        剧情自己播完回到地图，菱形消失。只认备战时脚本把它当成没弹窗，连点 10 次后停在弹窗上。
        """
        if self.popup() != 'talk':
            return self.fight()
        logger.info('SideQuest: 对话类支线，点「对话」')
        self.tap(QUEST_TALK_POINT, 3)
        state, _ = self.wait_map(BATTLE_WAIT)
        # 没回到地图（还停在弹窗上）就交给调用方停止，不在这里反复点
        return 'talk' if state == 'map' else 'skip'

    def fight(self):
        """备战弹窗已出现：核对 SUB 标题后进入战斗，等回到地图。返回 'win' / 'defeat' / 'skip'。"""
        self.device.screenshot()
        stage = sub_stage(self.entry_ocr.ocr(self.device.image, threshold=.8)['text'])
        if stage is None:
            logger.warning('SideQuest: 备战标题不是 SUB 关卡，关闭弹窗')
            self.device.click(FIGHT_CLOSE)
            time.sleep(2)
            return 'skip'
        logger.info(f'SideQuest: 进入 SUB {stage[0]}-{stage[1]}')
        self.device.click(FIGHT)
        deadline = time.monotonic() + BATTLE_WAIT
        result, left = 'win', False
        while time.monotonic() < deadline:
            self.device.screenshot()
            if self.appear(OPERATION_FAILED, offset=10):
                result = 'defeat'
                self.device.click(END_FIGHTING)
            elif self.appear_then_click(END_FIGHTING, offset=30, interval=2):
                pass
            elif self.appear_then_click(FIGHT, offset=30, interval=5):
                pass
            elif self.appear_then_click(AUTO_SHOOT, offset=10, threshold=.9, interval=5) or \
                    self.appear_then_click(AUTO_BURST, offset=10, threshold=.9, interval=5):
                pass
            elif map_visible(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)):
                # 实录 2026-10-23（SUB 23-8）：点进入战斗后加载前还有一瞬地图，必须先离开过地图才算打完
                if left:
                    return result
            elif not self.appear(FIGHT, offset=30):
                # 备战弹窗也会盖住地图锚点，弹窗和地图都不在才算进了战斗
                left = True
            time.sleep(1)
        raise RuntimeError('SUB 关卡战斗超时')

    def clear_map(self):
        """在当前章节地图上打完所有支线「!」目标，返回 (胜场, 停止原因)。"""
        wins = misses = sweep = 0
        last = None
        for _ in range(MAX_FIGHTS * 3):
            state, image = self.wait_map()
            if state is None:
                return wins, '未回到章节地图'
            if state == 'entry':
                entered = self.engage()
                if entered in ('win', 'talk'):
                    wins, misses = wins + (entered == 'win'), 0
                    continue
                return wins, '支线战斗失败' if entered == 'defeat' else '弹窗无法处理（不是 SUB 关卡或对话未进入）'
            markers = quest_markers(image)
            if not markers:
                time.sleep(2)
                image = self.bgr()
                markers = quest_markers(image)
            if not markers:
                if sweep >= len(SWEEP):
                    return wins, None
                start, end = SWEEP[sweep]
                sweep += 1
                self.device.swipe(start, end, method='swipe', handle_control_check=False)
                time.sleep(1.5)
                continue
            sweep = 0
            point = markers[0]
            if last is not None and abs(point[0] - last[0]) <= 40 and abs(point[1] - last[1]) <= 40 \
                    and not any(x1 <= point[0] < x2 and y1 <= point[1] + ENEMY_BELOW_MARKER < y2
                                for x1, y1, x2, y2 in UNSAFE):
                # 实录 2026-10-23（SUB 23-1）：同一菱形点完没弹窗，小队就站在菱形下；改点下方的敌人
                point = (point[0], point[1] + ENEMY_BELOW_MARKER)
            # 点过下方后下一轮回到点菱形本身，两种点法交替
            last = markers[0] if point == markers[0] else None
            self.tap(point, 1)
            # 场内菱形：小队寻路走过去后弹备战；边缘菱形：只移镜头，不会弹窗。
            # 两者在画面上分不清，所以只等一小段，没弹窗就重新找菱形再点（重点同一个目标只是续上寻路）。
            deadline = time.monotonic() + ENTRY_WAIT
            entered = None
            while time.monotonic() < deadline:
                if self.popup() is not None:
                    entered = self.engage()
                    break
                time.sleep(1)
            if entered is not None:
                last = None
            if entered == 'talk':
                misses = 0
            elif entered == 'win':
                wins, misses = wins + 1, 0
                if wins >= MAX_FIGHTS:
                    return wins, None
            elif entered == 'defeat':
                return wins, '支线战斗失败'
            elif entered == 'skip':
                return wins, '弹窗无法处理（不是 SUB 关卡或对话未进入）'
            else:
                misses += 1
                if misses >= MAX_MISSES:
                    return wins, '点支线标记后多次未弹出备战（可能道路未打通）'
        return wins, None

    def back_to_lobby(self):
        """关掉地图上残留的弹窗后点 HOME 回大厅。

        实录 2026-10-23：对话弹窗盖住地图时直接 ui_ensure 认不出页面，抛 GamePageUnknownError。
        """
        for _ in range(3):
            kind = self.popup()
            if kind == 'fight':
                self.device.click(FIGHT_CLOSE)
            elif kind == 'talk':
                self.tap(TALK_CLOSE, 0)
            else:
                break
            time.sleep(2)
        if map_visible(self.bgr()):
            self.tap(MAP_HOME, 5)
        self.ui_ensure(page_main)

    def open(self):
        # 上次中断时可能停在章节地图或支线弹窗上，NKAS 页面识别认不出，先自己回大厅
        if self.popup() is not None or map_visible(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)):
            self.back_to_lobby()
        return super().open()

    def find_chat(self, wanted):
        """从列表顶部往下找第一个满足 wanted(标题截图) 的对话行，返回 (画面, 行 y)。"""
        image = self.list_image()
        for _ in range(LIST_SCROLL_ROUNDS):
            image = self.scroll_list(image, down=False)
            if image is None:
                break
        image = self.bgr()
        for _ in range(LIST_SCROLL_ROUNDS):
            for y in chat_rows(image):
                if wanted(self.row_identity(image, y)):
                    return image, y
            image = self.scroll_list(image, down=True)
            if image is None:
                break
        return None, None

    def handle_chat(self):
        """在打开的对话里领取「可完成」委托并读完后续对话；返回 (「直接前往」位置或 None, 领取数)。

        领取后先弹 REWARD，点掉回到对话，对话会接着发礼物和新消息（可能带出下一个委托）。
        实录 2026-10-23：西里尔【第18章】收集地面上的5朵花 1/1 →「可完成」领 30 钻后继续出「… ▼」。
        """
        claimed = 0
        for _ in range(6):
            chat = self.wait_view('chat', 10)
            if chat is None:
                return None, claimed
            done = quest_done(chat)
            if done is not None:
                self.tap(done, 3)
                claimed += 1
                for _ in range(5):
                    if self.wait_view('chat', 3) is not None:
                        break
                    self.tap(REWARD_CLOSE, 2)
                self.read_chat()
                continue
            go = quest_go(chat)
            if go is not None:
                return go, claimed
            if typing(chat):
                self.read_chat()
                continue
            return None, claimed
        return None, claimed

    def run_quests(self):
        """逐个对话领取已完成委托、前往未完成委托打 SUB；返回 (前往委托数, SUB 胜场, 领取数, 问题列表)。"""
        seen, quests, wins, claimed, problems = [], 0, 0, 0, []
        for _ in range(MAX_CHATS):
            image, y = self.find_chat(lambda title: not any(same_chat(title, old) for old in seen))
            if image is None:
                break
            title = self.row_identity(image, y)
            seen.append(title)
            self.tap((360, y), 2.5)
            go, count = self.handle_chat()
            claimed += count
            if go is None:
                self.tap(BACK, 2)
                continue
            quests += 1
            card = self.card_text()
            self.tap(go, 3)
            won, problem = self.clear_map()
            wins += won
            logger.info(f'SideQuest: 委托 {quests}「{card}」（{title["text"]}）打完 {won} 个 SUB 关卡'
                        + (f'，停止原因：{problem}' if problem else ''))
            if problem or not won:
                path = self.save_evidence(f'quest{quests}')
                if path is not None:
                    logger.info(f'SideQuest: 已存画面 {path}')
            self.back_to_lobby()
            if problem:
                problems.append(problem)
                if problem == '支线战斗失败':
                    break
            if not self.open():
                problems.append('回大厅后未能重新打开即时通讯')
                break
            # 回到同一个对话领取刚打完的委托（列表按最新对话排序，位置会变，按标题找）
            image, y = self.find_chat(lambda current: same_chat(current, title))
            if image is not None:
                self.tap((360, y), 2.5)
                _, count = self.handle_chat()
                claimed += count
                self.tap(BACK, 2)
            else:
                logger.warning(f'SideQuest: 没找回对话「{title["text"]}」领取委托奖励')
                problems.append('委托打完后没找回对话领取奖励')
        return quests, wins, claimed, problems

    def run(self):
        logger.hr('SideQuest', 1)
        try:
            if not self.open():
                report('skipped', '大厅没有找到即时通讯图标或列表未打开')
            else:
                quests, wins, claimed, problems = self.run_quests()
                status = 'partial' if problems else 'success'
                report(status, f'章节支线：前往 {quests} 个委托，打赢 {wins} 个 SUB 关卡，领取 {claimed} 个委托奖励',
                       '；'.join(dict.fromkeys(problems)))
                if view(self.bgr()) in ('list', 'chat'):
                    self.tap(CLOSE, 2)
        except Exception as e:
            report('failed', '章节支线异常结束', type(e).__name__, retryable=True)
            logger.error(e)
            self.back_to_lobby()
        self.ui_ensure(page_main)
        self.config.task_delay(schedule=True)
