"""独立推图任务：选择章节、进入主线并将实际结果写入简报。"""
import re

import cv2

from module.base.timer import Timer
from module.base.watchdog import WaitGuard, save_wait_image
from module.campaign.campaign import Campaign
from module.campaign.report import GOAL_REASONS
from module.campaign.vision.chapter import chapter_identity
from module.exception import GameStuckError, RequestHumanTakeover
from module.execution.context import report
from module.logger import logger
from module.ocr.ocr import Ocr
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

    def ensure_main_map(self):
        """经大厅的主线任务箭头进入地图，复位之前的镜头视野。

        主线任务引导使用 720×1280 客户区，确认大厅及引导文字后才点击。
        进入章节地图受 90 秒时限保护；章节选择随后在横屏尺寸单独核对。
        """
        self.device.target_resolution = (720, 1280)
        self.device.ensure_resolution(self.config.PCClient_ScreenNumber, 720, 1280,
                                      position=self.config.PCClient_GameWindowPosition)
        self.device.check_resolution(720, 1280)
        self.device.screenshot()
        self.ui_ensure(page_main)
        self.device.screenshot()
        current = self.quest_stage()
        if current is None:
            report('blocked', '无法确认当前主线任务引导', '请确认主线任务引导可见后重试，已保存当前画面')
            return False
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
        return sum(enemy.get('status') == 'cleared'
                   for enemy in getattr(getattr(campaign, 'state', None), 'enemies', {}).values())

    def run(self):
        """执行主线推图，定位失败后最多两次经大厅复位镜头并继续当前进度。

        每次重新进入均核对用户选择的章节，当前进度模式沿用最新主线进度。
        首次恢复仍无新清除结果时停止；战败、窗口变化等情况不自动恢复。
        全部段的确认清除数合并进简报，结束后恢复日常竖屏并仅延后本任务。
        """
        if self.config.Client_Platform != 'win' or self.config.CampaignMap_MapType != 'main':
            report('blocked', '当前推图仅支持 PC 主线地图', '请选择 PC 实例与主线地图')
            return
        if not self.runtime_ready():
            self.config.task_delay(schedule=True)
            return
        try:
            target = chapter_number(self.config.CampaignMap_Chapter)
        except ValueError as exc:
            report('blocked', '开始章节无效', str(exc))
            self.config.task_delay(schedule=True)
            return
        total_cleared = 0
        try:
            for recovery in range(3):
                if not self.ensure_main_map():
                    if total_cleared:
                        report('partial', f'本轮已确认清除 {total_cleared} 个敌人，当前入口需要处理')
                    return
                self.device.target_resolution = Campaign.RESOLUTION
                self.device.ensure_resolution(self.config.PCClient_ScreenNumber, *Campaign.RESOLUTION,
                                              position=self.config.PCClient_GameWindowPosition)
                self.device.check_resolution(*Campaign.RESOLUTION)
                campaign = Campaign(self.config, device=self.device)
                if not self.select_chapter(campaign, target):
                    if total_cleared:
                        report('partial', f'本轮已确认清除 {total_cleared} 个敌人，章节入口需要处理')
                    return
                try:
                    campaign.run()
                except RequestHumanTakeover as exc:
                    cleared = self.cleared_count(campaign)
                    total_cleared += cleared
                    result = getattr(campaign, 'last_result', {}) or {}
                    code = result.get('reason') or (result.get('goal') or {}).get('reason')
                    detail = str(exc.__cause__ or exc)
                    reason = {'tick_budget': '本轮执行次数达到上限',
                              'scouting_incomplete': '前置道路无法确认'}.get(
                        code, GOAL_REASONS.get(code) or detail
                        or '请查看日志与截图，确认当前地图、道路或战斗状态')
                    localized = code == 'localization_unavailable' or any(
                        word in detail for word in ('定位', '配准', '小地图'))
                    if localized and recovery < 2 and (recovery == 0 or cleared > 0):
                        save_wait_image(self.device, f'AutoPush_recovery_{recovery + 1}')
                        logger.warning(f'地图定位恢复 {recovery + 1}/2：本段确认清除 {cleared} 个敌人，'
                                       f'经大厅重新进入当前主线；原因：{reason}')
                        continue
                    report('partial' if total_cleared else 'blocked',
                           f'本轮已确认清除 {total_cleared} 个敌人，推图已停止'
                           if total_cleared else '主线推图需要处理', reason)
                    return
                total_cleared += self.cleared_count(campaign)
                report('partial', f'本轮已确认清除 {total_cleared} 个敌人',
                       '本轮有限执行已结束，请查看推图信息与日志确认后续进度')
                return
        except (RequestHumanTakeover, GameStuckError) as exc:
            report('failed' if isinstance(exc, GameStuckError) else 'partial' if total_cleared else 'blocked',
                   f'本轮已确认清除 {total_cleared} 个敌人，推图已停止' if total_cleared else '主线推图需要处理',
                   str(exc.__cause__ or exc) or '入口或地图状态需要处理，详情见日志与截图')
        finally:
            self.config.task_delay(schedule=True)
            self.device.target_resolution = (720, 1280)
            self.device.ensure_resolution(self.config.PCClient_ScreenNumber, 720, 1280,
                                          position=self.config.PCClient_GameWindowPosition)
            self.device.check_resolution(720, 1280)
