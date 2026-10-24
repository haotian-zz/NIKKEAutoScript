"""独立推图任务：选择章节、进入主线并将实际结果写入简报。"""
import re
import time

import cv2
import psutil

from module.base.timer import Timer
from module.base.watchdog import WaitGuard, save_wait_image
from module.campaign.campaign import Campaign
from module.campaign.chapter_flow import ChapterOverlayFlow
from module.campaign.navigation.controls import battle_popup_score
from module.campaign.session import ENTRY_CLOSE_POINT, CampaignSession
from module.campaign.report import GOAL_REASONS
from module.campaign.vision import battle
from module.campaign.vision.chapter import chapter_identity
from module.campaign.vision.chapter_completion import overlay_anchor
from module.campaign.vision import hard_list
from module.campaign.window import CampaignWindow
from module.campaign.visible_story import (DESTINATION_SCORE, destination_lobby_quest, destination_place,
                                           quest_jump_point, read_quest_text, return_lobby_quest)
from module.exception import GameStuckError, RequestHumanTakeover
from module.execution.context import report
from module.logger import logger
from module.ocr.ocr import Ocr
from module.simulation_room.assets import END_FIGHTING
from module.ui.assets import ARK_CHECK, MAIN_CHECK
from module.ui.page import page_main, page_outpost
from module.ui.ui import UI

# 切日后大厅可能连续弹出多个奖励／活动页，超过上限仍未见大厅则交人工确认。
LOBBY_POPUP_LIMIT = 5
# 大厅任务仍为“返回大厅”时等待剧情触发及任务更新的上限（秒）。
LOBBY_RETURN_WAIT = 30.
# 前往【某处】任务：单次点击后等待到达并更新任务的上限（秒）。
DESTINATION_QUEST_WAIT = 30.
# 大厅任务行右侧的任务箭头区域（720×1280）。
LOBBY_JUMP_AREA = (660, 108, 710, 149)
# 方舟页里任务目的地卡片的地名区域（720×1280）：点任务箭头后游戏把目的地卡片高亮放在中间。
ARK_DESTINATION_LABEL = (292, 600, 402, 656)


def chapter_number(value):
    """解析当前进度或 1 至 50 章的选择，拒绝无效编号以免启动后误选。"""
    text = str(value).strip()
    if text == 'current':
        return None
    if re.fullmatch(r'[1-9]\d?', text) and 1 <= int(text) <= 50:
        return int(text)
    raise ValueError('请选择当前主线进度或第 1 至 50 章')


class AutoPush(UI):
    """复用主线引导和上游执行器，保留解锁限制、有限执行与日常竖屏约定。"""
    def quest_stage(self):
        """从竖屏主线读取具体关卡，并独立保留已核对的非数字前往地面出击指令。

        固定单行识别优先，低分保留旧大区域OCR；两者均只保留0.9以上文字。
        出击须为完整指令，不将其虚构成下一关编号；实录 2026-10-21 前哨基地任务完成后只显示「【出击】」。
        """
        reader = Ocr((390, 65, 710, 185), name='CAMPAIGN_MAIN_QUEST',
                     model_type=self.config.Optimization_OcrModelType)
        text = read_quest_text(self.device.image, reader)
        if text is None:
            # 「前往【方舟】的【市内】」这类长行分数偏低；只接受整行前往格式，稳定窗口另要求两帧相同。
            loose = read_quest_text(self.device.image, reader, minimum=DESTINATION_SCORE)
            text = loose if destination_lobby_quest(loose) else None
        if text is None:
            text = reader.ocr(self.device.image, threshold=.9)['text']
        self.lobby_sortie_instruction = bool(re.fullmatch(r'(?:前往地面)?(?:【出[击擊]】|\[出[击擊]\])',
                                                         re.sub(r'\s+', '', text)))
        self.lobby_return_pending = return_lobby_quest(text)
        self.lobby_destination_instruction = destination_lobby_quest(text)
        self.lobby_quest_text = text
        match = re.search(r'(?:通关|CLEAR).*?([1-9]\d?)\s*[-－]\s*([1-9]\d?)', text, re.I)
        return tuple(map(int, match.groups())) if match else None

    def lobby_popup(self):
        """大厅首屏偶发的每日奖励、登录活动、礼包和公告浮层：只用既有模板／文字处理器，返回是否处理。

        实录 2026-10-21 切日后连续出现：两次「30天成长补给品每日奖励」，随后是活动登录奖励页
        （全部领取并返回）。它们遮住主线任务导致入口拒绝；登录奖励沿用日常任务的领取逻辑。
        """
        return bool(self.handle_reward(interval=1) or self.handle_login_reward() or self.handle_paid_gift()
                    or self.handle_event_popup() or self.handle_announcement())

    def lobby_anchor(self, popups):
        """等待大厅锚点；被已知奖励／公告浮层遮住时按模板关闭，返回(是否确认, 已处理浮层数)。

        调用前已确认过大厅，浮层淡入淡出期间锚点短暂缺失只做有界观察（约3秒），不点击未知画面。
        """
        for _ in range(10):
            self.device.screenshot()
            if self.appear(MAIN_CHECK, offset=30):
                return True, popups
            if popups < LOBBY_POPUP_LIMIT and self.lobby_popup():
                popups += 1
                logger.info(f'大厅出现奖励或公告浮层，已按既有模板关闭（{popups}/{LOBBY_POPUP_LIMIT}），重新确认主线任务')
                self.device.sleep(1.)
                continue
            self.device.sleep(.3)
        return False, popups

    def quest_identity(self, current):
        """稳定窗口比较用的任务身份：出击、前往前哨基地两种非数字指令各自独立，其余为关卡编号。"""
        if getattr(self, 'lobby_sortie_instruction', False):
            return ('sortie',)
        if getattr(self, 'lobby_destination_instruction', False):
            return ('destination', getattr(self, 'lobby_quest_text', None))
        return current

    def visit_quest_destination(self):
        """完成“前往【某处】”任务：只点一次连续两帧一致的任务箭头，由游戏带到目的地触发剧情。

        实录 2026-10-21（21 章通关）：大厅任务变为前往前哨基地，点击箭头进入基地即提示完成，
        约十秒后任务改为出击。只在任务栏位置已知的大厅／前哨基地／方舟页读取新任务；三十秒内未确认
        也不再重复点击，返回None交由调用方回大厅重新核对。未发送点击返回False。
        实录 2026-10-23（23 章通关）：「前往【方舟】的【市内】」点箭头后到方舟页，目的地卡片高亮在中间，
        点卡片后加载片刻任务变为「前往【方舟】的【简报室】」，再点一次卡片后变为「通关【24-1】」。
        方舟页上卡片地名与当前任务地名一致才点，每个地名只点一次；每点一次重新计时。
        """
        first = quest_jump_point(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR), LOBBY_JUMP_AREA)
        self.device.sleep(.3)
        self.device.screenshot()
        second = quest_jump_point(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR), LOBBY_JUMP_AREA)
        if (first is None or second is None or sum((a-b)**2 for a, b in zip(first, second)) > 25.
                or not self.appear(MAIN_CHECK, offset=30)):
            return False
        destination = self.lobby_quest_text
        logger.info(f'大厅任务为 {destination!r}：点击任务箭头，由游戏带到目的地触发剧情')
        self.device.click_xy(*second)
        visited = set()
        deadline = time.monotonic() + DESTINATION_QUEST_WAIT
        while time.monotonic() < deadline:
            self.device.sleep(1.)
            self.device.screenshot()
            ark = self.appear(ARK_CHECK, offset=30)
            if not (ark or self.appear(page_outpost.check_button, offset=30) or self.appear(MAIN_CHECK, offset=30)):
                continue
            self.quest_stage()
            if getattr(self, 'lobby_quest_text', None) and not self.lobby_destination_instruction:
                logger.info(f'{destination!r} 任务已完成，新任务：{self.lobby_quest_text!r}')
                return True
            place = destination_place(getattr(self, 'lobby_quest_text', None))
            if not ark or place is None or place in visited:
                continue
            point = self.ark_destination_card(place)
            if point is None:
                continue
            visited.add(place)
            logger.info(f'方舟页目的地卡片 {place!r} 与任务一致，点击进入')
            self.device.click_xy(*point)
            deadline = time.monotonic() + DESTINATION_QUEST_WAIT
        logger.info(f'{destination!r} 点击后{DESTINATION_QUEST_WAIT:.0f}秒内未读到新任务，回大厅重新核对')
        return None

    def ark_destination_card(self, place):
        """方舟页中间卡片的地名连续两帧都等于任务地名时返回卡片中心，否则None；不按位置盲点。"""
        x1, y1, x2, y2 = ARK_DESTINATION_LABEL
        reader = Ocr(ARK_DESTINATION_LABEL, name='ARK_DESTINATION_LABEL',
                     model_type=self.config.Optimization_OcrModelType)
        for index in range(2):
            if index:
                self.device.sleep(.3)
                self.device.screenshot()
                if not self.appear(ARK_CHECK, offset=30):
                    return None
            text = re.sub(r'\s+', '', reader.ocr(self.device.image, threshold=.9)['text'] or '')
            if text != place:
                return None
        return (x1+x2)/2, (y1+y2)/2

    def confirm_lobby_quest(self):
        """大厅首次显示可能尚在淡入，四秒内等待两帧相同有效任务，不以空OCR立即终止。

        每帧重核大厅锚点，数字任务与完整出击指令分别比较；没有确认时不发送入口点击。
        锚点被已知浮层遮住时最多处理五次，关闭后重新开始稳定窗口。
        """
        popups = 0
        while True:
            seen, popups = self.lobby_anchor(popups)
            if not seen:
                return None
            current = self.quest_stage()
            previous = self.quest_identity(current)
            # 首次OCR可能初始化模型；稳定窗口从组件已可读取后起算，不能读对首帧却无第二帧机会。
            started = time.monotonic()
            deadline = started + 4.
            restart = False
            while time.monotonic() < deadline:
                self.device.sleep(.3)
                self.device.screenshot()
                if not self.appear(MAIN_CHECK, offset=30):
                    restart = popups < LOBBY_POPUP_LIMIT and self.lobby_popup()
                    break
                current = self.quest_stage()
                sortie = getattr(self, 'lobby_sortie_instruction', False)
                identity = self.quest_identity(current)
                if identity is not None and identity == previous:
                    return current, sortie
                if getattr(self, 'lobby_return_pending', False):
                    # 实录 2026-10-21（21 章通关）：回到大厅后任务栏仍是“返回大厅”，几秒后才自动播剧情
                    # （截图时自动跳过）再更新任务；继续等待，最多 LOBBY_RETURN_WAIT 秒。
                    deadline = max(deadline, min(started+LOBBY_RETURN_WAIT, time.monotonic()+4.))
                previous = identity
            if not restart:
                return None
            popups += 1
            logger.info(f'大厅出现奖励或公告浮层，已按既有模板关闭（{popups}/{LOBBY_POPUP_LIMIT}），重新确认主线任务')
            self.device.sleep(1.)

    def recover_settlement(self):
        """继续上次中断后遗留的竖屏结算，不将旧胜利计入本次推图。

        仅点击现有 END_FIGHTING 模板确认的继续提示，最多等待二十秒；不启动或重试任何战斗。
        """
        guard = WaitGuard(self.device, '清理已有战斗结算', timeout=20, warn_after=8)
        while True:
            self.device.screenshot()
            guard.check()
            if not self.appear(END_FIGHTING, offset=30, threshold=.85):
                return
            if self.appear_then_click(END_FIGHTING, offset=30, threshold=.85, interval=1):
                logger.info('推图入口已继续已有战斗结算；该结果不计入本次胜利。')
            self.device.sleep(.5)

    def ensure_main_map(self):
        """经大厅的主线任务箭头进入地图，复位之前的镜头视野。

        主线任务引导使用 720×1280 客户区，确认大厅及引导文字后才点击。
        进入章节地图受 90 秒时限保护；章节选择随后在横屏尺寸单独核对。
        """
        if self.recover_chapter_overlays():
            self.entry_stage = None
            return True
        self.device.target_resolution = (720, 1280)
        self.device.ensure_resolution(self.config.PCClient_ScreenNumber, 720, 1280,
                                      position=self.config.PCClient_GameWindowPosition, interval=.15, stable_time=1.)
        self.device.check_resolution(720, 1280)
        self.recover_settlement()
        self.device.screenshot()
        self.ui_ensure(page_main)
        entry = self.confirm_lobby_quest()
        if entry is None:
            report('blocked', '无法确认当前主线任务引导', '请确认主线任务引导可见后重试，已保存当前画面')
            return False
        if getattr(self, 'lobby_destination_instruction', False):
            if self.visit_quest_destination() is False:
                report('blocked', '前往任务箭头未连续确认', '未发送任务点击，已保存当前画面')
                return False
            self.ui_ensure(page_main)
            entry = self.confirm_lobby_quest()
            if entry is None or self.lobby_destination_instruction:
                report('blocked', '前往任务未完成', '单次点击任务箭头后回到大厅，任务仍未更新；不重复点击，已保存当前画面')
                return False
        current, sortie = entry
        self.entry_stage = current
        point = (674, 126)
        if sortie:
            point = quest_jump_point(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR), LOBBY_JUMP_AREA)
            self.device.sleep(.3)
            self.device.screenshot()
            if (point is None or not self.appear(MAIN_CHECK, offset=30) or self.quest_stage() is not None
                    or not self.lobby_sortie_instruction):
                report('blocked', '前往地面出击指令未连续确认', '未发送任务点击，旧章待办保留')
                return False
            fresh_point = quest_jump_point(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR), LOBBY_JUMP_AREA)
            if fresh_point is None or sum((a-b)**2 for a, b in zip(point, fresh_point)) > 25.:
                report('blocked', '大厅任务箭头未连续确认', '未发送任务点击，旧章待办保留')
                return False
            point = fresh_point
            logger.info('通过已连续核验的前往地面出击指令进入地图，不假定下一章或关卡编号')
        else:
            logger.info(f'通过主线任务 {current[0]}-{current[1]} 进入地图')
        self.device.click_xy(*point)
        guard = WaitGuard(self.device, '进入主线地图', timeout=90, warn_after=30)
        timer = Timer(8 if sortie else 2).start()
        while True:
            self.device.screenshot()
            guard.check()
            text = Ocr((490, 1125, 700, 1265), name='CAMPAIGN_MAP_ENTRY',
                       model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
            if 'NORMAL' in text.upper():
                return True
            if sortie:
                from module.ui.page import page_campaign_select, page_surface
                if self.appear(page_campaign_select.check_button, offset=30):
                    self.ui_ensure(page_surface)
                    continue
            if timer.reached() and self.appear(MAIN_CHECK, offset=30):
                if sortie:
                    report('blocked', '出击任务单次点击后仍停在大厅', '不重复点击，保留当前任务及旧章待办')
                    return False
                self.device.click_xy(674, 126)
                timer.reset()
            self.ui_additional()

    def recover_chapter_overlays(self):
        """重启入口先处理遗留的已知章节通知及列表，确认仍在同一普通章后直接续用地图。

        先统一1776×999，普通界面仅做廉价锚点检查；没有浮层则返回False走既有大厅入口。
        有浮层时连续核对真实章号后才建正式会话；处理后两帧同章地图可见才返回True。
        四十五秒仍未知则停止，清理簿不改，旧胜利不计入本轮，也不从大厅任务猜章号。
        """
        self.device.target_resolution = Campaign.RESOLUTION
        self.device.ensure_resolution(self.config.PCClient_ScreenNumber, *Campaign.RESOLUTION,
                                      position=self.config.PCClient_GameWindowPosition, interval=.15, stable_time=1.)
        self.device.check_resolution(*Campaign.RESOLUTION)
        self.device.screenshot()
        field = cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)
        # 实录 2026-10-21：上一轮点中敌人后中断，备战弹窗留在地图上，大厅页面识别认不出直接停止。
        # 只点关闭、绝不开战；最多关两次，仍在就交人工，避免与不认识的弹窗来回点击。
        for _ in range(2):
            if battle_popup_score(field) <= .8:
                break
            logger.info('地图上残留战斗准备弹窗：仅点击关闭，不进入战斗。')
            self.device.click_xy(*ENTRY_CLOSE_POINT)
            self.device.sleep(1.5)
            self.device.screenshot()
            field = cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)
        else:
            if battle_popup_score(field) > .8:
                raise RequestHumanTakeover('战斗准备弹窗关闭两次仍在，未进入战斗。')
        if overlay_anchor(field) is None:
            return False
        owner = Campaign(self.config, device=self.device)
        win = CampaignWindow(owner)
        win.focus()
        first = chapter_identity(field, owner.chapter_ocr)
        fresh = win.client()
        second = chapter_identity(fresh, owner.chapter_ocr)
        if first is None or first != second:
            raise RequestHumanTakeover('遗留章节浮层下的地图身份未连续确认，未发送处理点击。')
        owner.current_chapter, owner.current_difficulty = first
        session = CampaignSession(owner, navigation=False)
        guard = WaitGuard(self.device, '恢复遗留章节通知', timeout=45, warn_after=15)
        ready = 0
        try:
            session.start()
            flow = ChapterOverlayFlow(session, owner.chapter_ocr)
            while True:
                guard.check()
                field = session.win.client()
                if flow.handle(field):
                    ready = 0
                    continue
                if battle.field_visible(field) and chapter_identity(field, owner.chapter_ocr) == first:
                    ready += 1
                    if ready >= 2:
                        logger.info(f'遗留章节界面已处理，连续确认仍在普通第{first[0]}章地图；旧章待办保留。')
                        return True
                else:
                    ready = 0
                session.pause(.35)
        finally:
            session.close()

    def map_difficulty_text(self):
        """竖屏地图右下角难度标签的文字（NORMAL / HARD），读不到返回空串。"""
        return Ocr((490, 1125, 700, 1265), name='CAMPAIGN_MAP_ENTRY',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text'].upper()

    def enter_hard(self, target=None):
        """竖屏地图上切到困难难度：点难度标签开章节列表 → 困难页签 → 最下面的已解锁卡片。

        困难页自动滚到当前进度，最下面的已解锁卡片就是困难最新章节；具体章节号随后在横屏由
        select_chapter 按 HARD 身份核对并用左右箭头切换。每步都等画面确认，不靠点击次数。
        已在困难地图且指定了章节时直接交给 select_chapter；「当前进度」总是经列表选最新章，
        因为停在旧困难章（例如刚通关、点了下一章控件后还在旧章）时地图本身不代表最新进度。
        实录 2026-10-23：普通 23 章地图 → 困难页（第 3 章 COMPLETE、第 4 章 0/18）→ 点第 4 章进入 HARD 04；
        困难 4 章通关浮层点「CHAPTER 05」后弹出的是暗红底章节列表，旧锚点认不出。
        """
        self.device.target_resolution = (720, 1280)
        self.device.ensure_resolution(self.config.PCClient_ScreenNumber, 720, 1280,
                                      position=self.config.PCClient_GameWindowPosition, interval=.15, stable_time=1.)
        self.device.check_resolution(720, 1280)
        guard = WaitGuard(self.device, '切换困难难度', timeout=150, warn_after=30)
        step, timer, settle = 'tag', Timer(10).start(), None
        while True:
            self.device.screenshot()
            guard.check()
            image = cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR)
            if hard_list.list_visible(image):
                if step == 'card' and not settle.reached():
                    # 实录 2026-10-23（困难 5→6）：点卡片后列表淡出的过渡帧里页签红色认不出，
                    # 回头再点页签会把 step 改掉，地图出来读到 HARD 也不返回，直到超时。
                    pass
                elif not hard_list.hard_tab_active(image):
                    if step != 'tab' or timer.reached():
                        self.device.click_xy(*hard_list.HARD_TAB)
                        step = 'tab'
                        timer.reset()
                elif step != 'card' or settle.reached():
                    self.device.sleep(1)
                    self.device.screenshot()
                    cards = hard_list.unlocked_cards(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR))
                    if not cards:
                        save_wait_image(self.device, 'AutoPush_hard_no_chapter')
                        self.device.click_xy(*hard_list.LIST_CLOSE)
                        report('blocked', '困难难度没有已解锁的章节', '先通关对应普通章节解锁困难难度')
                        return False
                    logger.info(f'困难章节列表：已解锁卡片 {cards}，进入最下面一张')
                    self.device.click_xy(*cards[-1])
                    step = 'card'
                    timer, settle = Timer(90).start(), Timer(10).start()
            else:
                text = self.map_difficulty_text()
                if 'HARD' in text and (step == 'card' or target is not None):
                    logger.info('已进入困难难度地图')
                    return True
                if step == 'card' and timer.reached():
                    save_wait_image(self.device, 'AutoPush_hard_switch')
                    report('blocked', '未能切换到困难难度', '点了困难章节卡片后困难地图未确认，已保存当前画面')
                    return False
                # 点标签后列表没出来就再点一次，总时长由 WaitGuard 兜底
                if ('NORMAL' in text or 'HARD' in text) and (step == 'tag' or step == 'list' and timer.reached()):
                    self.device.click_xy(*hard_list.DIFFICULTY_TAG)
                    step = 'list'
                    timer.reset()
            self.device.sleep(.5)

    def select_chapter(self, campaign, target, difficulty='normal'):
        """通过横屏右下角相邻章节按钮逐章切换，每次均重新识别章节与难度。

        仅在 1776×999 画面发送输入，每步最多等待 30 秒，总计最多六分钟。
        不把点击次数当作选中成功；入口锁定或加载未确认时结束并报告当前章节。
        困难难度 target 为 None 时只核对当前地图确实是 HARD。
        """
        if target is None and difficulty == 'normal':
            return True
        guard = WaitGuard(self.device, '选择主线章节', timeout=360, warn_after=30)
        pending, timer = None, Timer(30)
        while True:
            self.device.screenshot()
            guard.check()
            if self.device.image.shape[:2] != (999, 1776):
                raise GameStuckError('选择主线章节时画面尺寸发生变化')
            identity = chapter_identity(cv2.cvtColor(self.device.image, cv2.COLOR_RGB2BGR), campaign.chapter_ocr)
            if identity is not None:
                current, observed = identity
                if observed != difficulty:
                    report('blocked', f'未确认{"困难" if difficulty == "hard" else "普通"}主线地图', '请检查地图难度后重试')
                    return False
                if target is None or current == target:
                    logger.info(f'已确认进入{"困难" if difficulty == "hard" else "普通"}主线第 {current} 章')
                    return True
                if pending is not None and current != pending:
                    pending = None
                if pending is None:
                    logger.info(f'选择主线章节：当前 {current}，目标 {target}')
                    self.device.click_xy(1551 if target < current else 1715, 940)
                    pending = current
                    timer.reset()
            if pending is not None and timer.reached():
                report('blocked', f'未能从第 {pending} 章切换到第 {target} 章',
                       '章节可能尚未解锁或切换未生效；已停止，未在其他章节开始战斗')
                return False
            self.device.sleep(0.5)

    def runtime_ready(self):
        """在导航前检查严重内存压力、ONNX依赖与模型，失败时保留明确可补做原因。

        系统可用内存低于512MiB时不初始化检测器或发送游戏操作；这是低余量保护，
        不是按总内存或显卡档次限制。读取资源状态失败时仍由原模型/输入闸门处理。
        """
        try:
            available = psutil.virtual_memory().available
        except OSError as exc:
            logger.warning(f'无法读取系统可用内存，保留既有组件检查：{exc}')
            available = None
        if available is not None and available < 512*1024**2:
            amount = max(0, int(available/1024**2))
            logger.warning(f'自动推图启动停止：系统可用内存仅{amount}MiB，未初始化检测模型或发送游戏操作。')
            report('blocked', '系统可用内存不足，自动推图暂未启动',
                   f'当前可用{amount}MiB，最低启动余量512MiB；释放内存后可重试，当前地图和任务待办保留')
            return False
        try:
            from module.campaign.vision.detection import configure_inference, detector
            configure_inference(self.config.config_name)
            detector()
            return True
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            logger.error(f'自动推图组件检查失败：{exc}')
            report('blocked', '推图识别组件未就绪',
                   '检查 requirements-yolo.txt 的依赖与模型文件后重试：' + str(exc))
            return False

    @staticmethod
    def cleared_count(campaign):
        """统计本段已通过胜利结算确认清除的敌人，不把移动成功当作通关。"""
        return (len(getattr(campaign, 'story_wins', ()))
                + sum(enemy.get('status') == 'cleared'
                      for enemy in getattr(getattr(campaign, 'state', None), 'enemies', {}).values()))

    def renew_progress_guard(self):
        """只在真实胜利或主线目标前进后续期总时限，持续模式不因固定总时长中断。

        单目标操作预算、剧情跳过时限与每段执行预算仍各自生效；镜头移动不算进展。
        """
        guard = getattr(self.device, '_execution_guard', None)
        if guard is not None:
            guard.progress()

    def run(self):
        """执行单轮或从当前主线持续推进，保留定位恢复、失败终止与竖屏收尾。

        持续模式只在一段有确认胜利，或主线目标前进后继续下一段，不把预算当作永久终点。
        无目标时仅重新核对一次大厅任务；目标未变即停止。定位最多恢复两次，首次恢复
        后仍无新胜利则停止；战败、单目标预算耗尽和未知状态不会作为续跑理由。
        """
        if self.config.Client_Platform != 'win' or self.config.CampaignMap_MapType != 'main':
            report('blocked', '当前推图仅支持 PC 主线地图', '请选择 PC 实例与主线地图')
            return
        if not self.runtime_ready():
            self.config.task_delay(schedule=True)
            return
        try:
            mode = getattr(self.config, 'CampaignMap_RunMode', 'single')
            if mode not in ('single', 'continuous'):
                raise ValueError('请选择单轮推图或持续推图')
            continuous = mode == 'continuous'
            target = None if continuous else chapter_number(self.config.CampaignMap_Chapter)
            difficulty = getattr(self.config, 'CampaignMap_Difficulty', 'normal')
            if difficulty not in ('normal', 'hard'):
                raise ValueError('请选择普通或困难难度')
        except ValueError as exc:
            report('blocked', '推图设置无效', str(exc))
            self.config.task_delay(schedule=True)
            return
        total_cleared, recovery, segment = 0, 0, 0
        require_advance = None
        require_story_return = None
        returned_chapters = set()
        resume_map = False
        try:
            while True:
                if not resume_map and not self.ensure_main_map():
                    if total_cleared:
                        report('partial', f'本轮已确认清除 {total_cleared} 个敌人，当前入口需要处理')
                    return
                # 大厅任务箭头只通往普通地图，困难难度在竖屏地图上经章节列表切换
                if not resume_map and difficulty == 'hard' and not self.enter_hard(target):
                    return
                entry = getattr(self, 'entry_stage', None)
                if require_story_return is not None:
                    origin_chapter, previous = require_story_return
                    sortie = getattr(self, 'lobby_sortie_instruction', False)
                    if (not sortie and (entry is None or previous is not None and entry <= previous
                            or previous is None and entry[0] <= origin_chapter)):
                        save_wait_image(self.device, 'AutoPush_story_return_unconfirmed')
                        report('partial' if total_cleared else 'blocked', '返回大厅后的主线任务尚未确认推进',
                               '保留旧章待办；未以点击任务箭头作为进展，也不重复返回大厅')
                        return
                    logger.info(f'大厅已确认返回任务后的新主线目标：{entry if entry is not None else "前往地面出击"}；'
                                '旧章待办保留')
                    self.renew_progress_guard()
                    require_story_return = None
                if require_advance is not None:
                    if entry is None or entry <= require_advance:
                        save_wait_image(self.device, 'AutoPush_no_progress')
                        report('partial' if total_cleared else 'blocked',
                               f'本轮已确认清除 {total_cleared} 个敌人，持续推图已停止',
                               f'没有可继续的地图目标，主线任务仍为 {require_advance[0]}-{require_advance[1]}')
                        return
                    logger.info(f'主线目标已前进：{require_advance} → {entry}，继续推图')
                    self.renew_progress_guard()
                    require_advance = None
                self.device.target_resolution = Campaign.RESOLUTION
                self.device.ensure_resolution(self.config.PCClient_ScreenNumber, *Campaign.RESOLUTION,
                                              position=self.config.PCClient_GameWindowPosition,
                                              interval=.15, stable_time=1.)
                self.device.check_resolution(*Campaign.RESOLUTION)
                campaign = Campaign(self.config, device=self.device)
                campaign.continuous = continuous
                campaign.story_first = True
                if not self.select_chapter(campaign, target, difficulty):
                    if total_cleared:
                        report('partial', f'本轮已确认清除 {total_cleared} 个敌人，章节入口需要处理')
                    return
                if continuous:
                    target = None
                segment += 1
                logger.info(f'推图第 {segment} 段开始；模式 {mode}，累计确认清除 {total_cleared} 个敌人')
                error = None
                try:
                    campaign.run()
                except RequestHumanTakeover as exc:
                    error = exc
                cleared = self.cleared_count(campaign)
                total_cleared += cleared
                result = getattr(campaign, 'last_result', {}) or {}
                code = (result.get('reason') or (result.get('goal') or {}).get('reason')
                        or (result.get('evidence') or {}).get('kind'))
                detail = str(error.__cause__ or error) if error is not None else ''
                reason = result.get('message') or {'tick_budget': '本段决策次数达到上限',
                          'segment_timeout': '本段执行时间达到上限',
                          'scouting_incomplete': '前置道路无法确认',
                          'chapter_changed': '已进入下一主线章节',
                          'ex_stage_skipped': '遇到 EX 关卡，已关闭弹窗，本模式不接管 EX 战斗',
                          'collectible_indicator': '遇到收集品提示，需要处理后继续'}.get(
                    code, GOAL_REASONS.get(code) or detail
                    or '请查看日志与截图，确认当前地图、道路或战斗状态')
                if code == 'chapter_cleared':
                    report('success', f'本章清怪巡检完成，本轮确认清除 {total_cleared} 个敌人', reason)
                    return
                if continuous and code == 'story_return_lobby':
                    origin_chapter = result.get('chapter')
                    if not isinstance(origin_chapter, int) or origin_chapter in returned_chapters:
                        report('partial' if total_cleared else 'blocked', '返回大厅任务重复或章节身份缺失，推图已停止',
                               '旧章待办保留，不重复发送返回操作')
                        return
                    returned_chapters.add(origin_chapter)
                    require_story_return = (origin_chapter, entry)
                    target = None
                    resume_map = False
                    logger.info('主线要求返回大厅；重新核对实际任务后继续当前主线，旧章可选内容保留待办')
                    continue
                if continuous and code in ('tick_budget', 'segment_timeout') and cleared:
                    logger.info(f'{reason}，本段确认清除 {cleared} 个敌人，继续当前主线进度')
                    self.renew_progress_guard()
                    resume_map = True
                    continue
                if continuous and code == 'chapter_changed':
                    self.renew_progress_guard()
                    resume_map = True
                    continue
                # 困难没有自己的主线任务，大厅任务不会因困难进度前进，不走「等任务前进再续跑」
                if continuous and code == 'no_reachable_target' and entry is not None and difficulty == 'normal':
                    require_advance = entry
                    resume_map = False
                    logger.info(f'{reason}，重新核对主线任务后决定是否继续')
                    continue
                localized = code == 'localization_unavailable' or any(
                    word in detail for word in ('定位', '配准', '小地图'))
                if localized and recovery < 2 and (recovery == 0 or cleared > 0):
                    recovery += 1
                    resume_map = False
                    save_wait_image(self.device, f'AutoPush_recovery_{recovery}')
                    logger.warning(f'地图定位恢复 {recovery}/2：本段确认清除 {cleared} 个敌人，'
                                   f'经大厅重新进入当前主线；原因：{reason}')
                    if cleared:
                        self.renew_progress_guard()
                    continue
                if error is not None:
                    report('partial' if total_cleared else 'failed' if code == 'battle_defeat' else 'blocked',
                           f'本轮已确认清除 {total_cleared} 个敌人，推图已停止'
                           if total_cleared else '主线推图需要处理', reason)
                    return
                report('partial' if total_cleared else 'blocked',
                       f'本轮已确认清除 {total_cleared} 个敌人，推图已停止',
                       reason if continuous else '本轮有限执行已结束，请查看推图信息与日志确认后续进度')
                return
        except (RequestHumanTakeover, GameStuckError) as exc:
            report('failed' if isinstance(exc, GameStuckError) else 'partial' if total_cleared else 'blocked',
                   f'本轮已确认清除 {total_cleared} 个敌人，推图已停止' if total_cleared else '主线推图需要处理',
                   str(exc.__cause__ or exc) or '入口或地图状态需要处理，详情见日志与截图')
        finally:
            self.config.task_delay(schedule=True)
            self.device.target_resolution = (720, 1280)
            self.device.ensure_resolution(self.config.PCClient_ScreenNumber, 720, 1280,
                                          position=self.config.PCClient_GameWindowPosition, interval=.15, stable_time=1.)
            self.device.check_resolution(720, 1280)
