import random
from functools import cached_property

from module.execution.context import report
from module.base.arena import ArenaBase
from module.base.timer import Timer
from module.champion_arena.assets import *
from module.logger import logger
from module.ocr.ocr import Digit
from module.ui.assets import ARENA_GOTO_CHAMPION_ARENA, CHAMPION_ARENA_CHECK
from module.ui.page import page_arena
from module.ui.ui import UI


class ChampionArenaIsUnavailable(Exception):
    pass


class ChampionArenaUnrecognized(Exception):
    pass


# 竞技场页上既没有可点入口、也没进入冠军竞技场、又不是已知赛季阶段的最长等待（秒）。
CHAMPION_ENTRY_SECONDS = 10


class ChampionArena(UI, ArenaBase):
    @cached_property
    def cheers(self):
        return [CHEER_SELECT_1, CHEER_SELECT_2]

    @cached_property
    def cheer_count(self) -> int:
        """获取左侧应援数值"""
        model_type = self.config.Optimization_OcrModelType
        CHEER_LEFT_COUNT = Digit(
            [CHEER_COUNT.area],
            name='CHEER_LEFT_COUNT',
            model_type=model_type,
            lang='num',
        )
        return int(CHEER_LEFT_COUNT.ocr(self.device.image)['text'])

    @property
    def cheer_select(self) -> int:
        """根据策略确定应援选择"""

        if self.config.CheerStrategy_Strategy == 'Most':
            return 0 if self.cheer_count > 50 else 1
        elif self.config.CheerStrategy_Strategy == 'Few':
            return 1 if self.cheer_count > 50 else 0
        elif self.config.CheerStrategy_Strategy == 'Random':
            return random.randint(0, 1)
        else:
            return 0

    def cheer(self, skip_first_screenshot=True):
        """应援"""
        logger.info('Open promotion or champion')
        click_timer = Timer(0.3)
        confirm_timer = Timer(3, count=3)

        # 打开晋级赛或者冠军争霸赛
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            # 晋级赛
            if click_timer.reached() and self.appear_then_click(PROMOTION, offset=10, interval=3):
                click_timer.reset()
                continue

            # 冠军争霸赛
            if click_timer.reached() and self.appear_then_click(CHAMPION, offset=10, interval=3):
                click_timer.reset()
                continue

            # 应援结果
            if (
                click_timer.reached()
                and self.appear(CHEER_RESULT, offset=10, static=False)
                and self.appear_then_click(CHEER_RESULT_NEXT, offset=10, interval=1, static=False)
            ):
                click_timer.reset()
                continue

            # 检查应援按钮
            if self.appear(PROMOTION_CHECK, offset=30) or self.appear(CHAMPION_CHECK, offset=30):
                if not confirm_timer.started():
                    confirm_timer.start()
                if confirm_timer.reached():
                    break
            else:
                confirm_timer.clear()

        # 检查应援按钮状态
        if not self.appear(CHEER_ENABLE, offset=10):
            logger.info('Cheer already done')
            raise ChampionArenaIsUnavailable

        logger.info('Cheer a nikke')
        cheer_done = False
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if (
                click_timer.reached()
                and (self.appear(PROMOTION_CHECK, offset=30) or self.appear(CHAMPION_CHECK, offset=30))
                and self.appear_then_click(CHEER_ENABLE, offset=10, interval=2)
            ):
                click_timer.reset()
                continue

            # 应援选择
            if (
                click_timer.reached()
                and self.appear(CHEER_CHECK, offset=10)
                and self.appear_then_click(self.cheers[self.cheer_select], threshold=10, interval=1)
            ):
                click_timer.reset()
                continue

            # 应援确认
            if (
                click_timer.reached()
                and self.appear(CHEER_CHECK, offset=10)
                and (not self.appear(self.cheers[0], threshold=10) or not self.appear(self.cheers[1], threshold=10))
                and self.appear_then_click(CHEER_CONFIRM, threshold=10, interval=1)
            ):
                cheer_done = True
                click_timer.reset()
                continue

            # 关闭
            if (
                cheer_done
                and self.appear(CHEER_CHECK, offset=10)
                and (not self.appear(self.cheers[0], threshold=10) or not self.appear(self.cheers[1], threshold=10))
                and self.appear_then_click(CHEER_CANCEL, threshold=10, interval=1)
            ):
                logger.info('Cheer done')
                break

    def ensure_into_champion_arena(self, skip_first_screenshot=True):
        logger.hr('CHAMPION ARENA START')

        # 实录 2026-10-22：入口显示“COMING SOON!”，不在已知阶段图里，原循环空转到900秒任务总时限。
        # 认不出入口状态时按有界等待结束，不为每种新文字加专门模板。
        entry_timer = Timer(CHAMPION_ENTRY_SECONDS, count=CHAMPION_ENTRY_SECONDS).start()
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if entry_timer.reached():
                logger.warning('Champion arena entry state not recognized')
                raise ChampionArenaUnrecognized

            if (
                self.appear(PROGRESS_NEXT_SEASON, offset=(30, 30))
                or self.appear(PROGRESS_TEAM, offset=(30, 30))
                or self.appear(PROGRESS_FIGHTING, offset=(30, 30))
                or self.appear(PROGRESS_RESULTING, offset=(30, 30))
                or self.appear(PROGRESS_END, offset=(30, 30))
                or self.appear(PROGRESS_COMING_SOON, offset=(30, 30))
            ):
                logger.warning('Champion arena in preparation')
                raise ChampionArenaIsUnavailable

            if self.appear_then_click(ARENA_GOTO_CHAMPION_ARENA, offset=30, interval=3):
                entry_timer.reset()
                continue

            if self.appear(CHAMPION_ARENA_CHECK, offset=30):
                break

    def run(self):
        self.ui_ensure(page_arena)
        try:
            self.ensure_into_champion_arena()
            self.cheer()
            report('success', '已检查冠军竞技场并处理当前可应援对局')
        except ChampionArenaIsUnavailable:
            report('skipped', '冠军竞技场当前没有可应援对局')
        except ChampionArenaUnrecognized:
            report('skipped', f'冠军竞技场入口{CHAMPION_ENTRY_SECONDS}秒内未出现可点入口或已知阶段（如尚未开放），已跳过')
        self.config.task_delay(schedule=True)
