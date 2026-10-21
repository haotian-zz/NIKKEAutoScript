"""独立推图任务：选择章节、进入主线并将实际结果写入简报。"""
import re

import cv2

from module.base.timer import Timer
from module.base.watchdog import WaitGuard, save_wait_image
from module.campaign.campaign import Campaign
from module.campaign.clearance import pending_chapter
from module.campaign.report import GOAL_REASONS
from module.campaign.vision.chapter import chapter_identity
from module.exception import GameStuckError, RequestHumanTakeover
from module.execution.context import report
from module.logger import logger
from module.ocr.ocr import Ocr
from module.simulation_room.assets import END_FIGHTING
from module.ui.assets import MAIN_CHECK
from module.ui.page import page_main
from module.ui.ui import UI


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
        """从竖屏右上角主线任务读取具体关卡，不把活动任务或章节号当作关卡。"""
        text = Ocr((390, 65, 710, 185), name='CAMPAIGN_MAIN_QUEST',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        match = re.search(r'(?:通关|CLEAR).*?([1-9]\d?)\s*[-－]\s*([1-9]\d?)', text, re.I)
        return tuple(map(int, match.groups())) if match else None

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
        self.device.target_resolution = (720, 1280)
        self.device.ensure_resolution(self.config.PCClient_ScreenNumber, 720, 1280,
                                      position=self.config.PCClient_GameWindowPosition, interval=.15, stable_time=1.)
        self.device.check_resolution(720, 1280)
        self.recover_settlement()
        self.device.screenshot()
        self.ui_ensure(page_main)
        self.device.screenshot()
        current = self.quest_stage()
        if current is None:
            report('blocked', '无法确认当前主线任务引导', '请确认主线任务引导可见后重试，已保存当前画面')
            return False
        self.entry_stage = current
        logger.info(f'通过主线任务 {current[0]}-{current[1]} 进入地图')
        self.device.click_xy(674, 126)
        guard = WaitGuard(self.device, '进入主线地图', timeout=90, warn_after=30)
        timer = Timer(2).start()
        while True:
            self.device.screenshot()
            guard.check()
            text = Ocr((490, 1125, 700, 1265), name='CAMPAIGN_MAP_ENTRY',
                       model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
            if 'NORMAL' in text.upper():
                return True
            if timer.reached() and self.appear(MAIN_CHECK, offset=30):
                self.device.click_xy(674, 126)
                timer.reset()
            self.ui_additional()

    def select_chapter(self, campaign, target):
        """通过横屏右下角相邻章节按钮逐章切换，每次均重新识别章节与普通难度。

        仅在 1776×999 画面发送输入，每步最多等待 30 秒，总计最多六分钟。
        不把点击次数当作选中成功；入口锁定或加载未确认时结束并报告当前章节。
        """
        if target is None:
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
                current, difficulty = identity
                if difficulty != 'normal':
                    report('blocked', '未确认普通主线地图', '请检查地图难度后重试')
                    return False
                if current == target:
                    logger.info(f'已确认进入普通主线第 {target} 章')
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
        """在导航前检查 ONNX 依赖与模型完整性，失败时写入可补做的待处理原因。"""
        try:
            from module.campaign.vision.detection import detector
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
            target = pending_chapter(self.config) if continuous else chapter_number(self.config.CampaignMap_Chapter)
        except ValueError as exc:
            report('blocked', '推图设置无效', str(exc))
            self.config.task_delay(schedule=True)
            return
        total_cleared, recovery, segment = 0, 0, 0
        require_advance = None
        resume_map = False
        try:
            while True:
                if not resume_map and not self.ensure_main_map():
                    if total_cleared:
                        report('partial', f'本轮已确认清除 {total_cleared} 个敌人，当前入口需要处理')
                    return
                entry = getattr(self, 'entry_stage', None)
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
                if not self.select_chapter(campaign, target):
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
                if continuous and code in ('tick_budget', 'segment_timeout') and cleared:
                    logger.info(f'{reason}，本段确认清除 {cleared} 个敌人，继续当前主线进度')
                    self.renew_progress_guard()
                    resume_map = True
                    continue
                if continuous and code == 'chapter_changed':
                    self.renew_progress_guard()
                    resume_map = True
                    continue
                if continuous and code == 'no_reachable_target' and entry is not None:
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
