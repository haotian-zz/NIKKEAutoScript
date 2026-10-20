import cv2

from module.execution.context import report
from module.base.timer import Timer
from module.handler.assets import CONFIRM_B
from module.logger import logger
from module.notify import handle_notify
from module.reward.assets import *
from module.ui.page import *
from module.ui.ui import UI


class NoRewards(Exception):
    pass


class Reward(UI):
    def receive_reward(self, skip_first_screenshot=True):
        logger.hr('Receive reward')
        confirm_timer = Timer(5, count=3).start()
        # Set click interval to 0.3, because game can't respond that fast.
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.handle_level_up():
                confirm_timer.reset()
                continue
            # 点击领取奖励
            if self.handle_reward(interval=1):
                confirm_timer.reset()
                continue

            if self.handle_paid_gift():
                confirm_timer.reset()
                continue
            # 判断是否有奖励可领
            if self.appear(NO_REWARDS_1, offset=(10, 10), interval=1) and confirm_timer.reached():
                logger.info('Reward done after check NO_REWARDS_1')
                break
            # 点击获得奖励
            if self.appear_then_click(RECEIVE, offset=(30, 30), interval=10):
                confirm_timer.reset()
                continue

            if confirm_timer.reached() and self.appear(EMPTY_CHECK, threshold=1.00):
                logger.info('Reward done after check EMPTY_CHECK')
                break

            if self.appear(MAIN_CHECK, offset=(10, 10)) and confirm_timer.reached():
                logger.info('Reward done after check MAIN_CHECK')
                break

        logger.info('Defence Reward have been received')
        return True

    def receive_social_point(self, skip_first_screenshot=True):
        logger.hr('Receive social point')
        confirm_timer = Timer(5, count=3).start()
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.appear_then_click(SEND_AND_RECEIVE, offset=(30, 30), interval=2):
                confirm_timer.reset()
                continue

            if self.appear_then_click(CONFIRM_B, offset=(30, 30), interval=1, static=False):
                confirm_timer.reset()
                continue

            if confirm_timer.reached():
                break

        logger.info('Social Point have been received')
        return True

    def receive_special_arena_point(self, skip_first_screenshot=True):
        logger.hr('Receive special arena point')
        confirm_timer = Timer(6, count=5).start()
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.appear_then_click(ARENA_GOTO_SPECIAL_ARENA, offset=(30, 30), interval=5, static=False):
                confirm_timer.reset()
                continue

            if self.appear_then_click(RECEIVE_SPECIAL_ARENA_POINT, offset=(30, 30), interval=5, static=False):
                confirm_timer.reset()
                continue

            if self.appear_then_click(REWARD_B, offset=(30, 30), interval=5, static=False):
                confirm_timer.reset()
                continue

            if self.appear(NO_REWARDS, offset=(5, 5), threshold=0.95, static=False):
                report('success', '特竞点数暂无可领取奖励')
                raise NoRewards

            elif self.appear(NO_REWARDS_2, offset=(5, 5), threshold=0.95, static=False):
                return True

            elif self.appear(NO_REWARDS_3, offset=(5, 5), threshold=0.95):
                return True

            if self.handle_reward(interval=1):
                logger.info('Special Arena Point have been received')
                report('success', '特竞点数已领取')
                raise NoRewards

            if confirm_timer.reached():
                report('unknown', '特竞点数领取结果未确认', retryable=True)
                raise NoRewards

        return True

    def receive_ranking(self, skip_first_screenshot=True):
        """通过排名图标领取奖励，并在入口或领取状态无法确认时限时退出。

        方舟侧栏的相邻入口可能同时显示红点，不能用大范围红点匹配结果作为点击位置。
        入口和奖励阶段分别限时，避免版面变化阻塞整轮日常。
        """
        logger.hr('Receive ranking reward')
        entry_timeout = Timer(20, count=20).start()
        confirm_timer = Timer(1, count=3)
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.appear(RANKING_CHECK, offset=10):
                if not confirm_timer.started():
                    confirm_timer.start()
                if confirm_timer.reached():
                    break
            else:
                confirm_timer.clear()

            if entry_timeout.reached():
                logger.warning('Ranking reward skipped: ranking page did not open within 20 seconds')
                return False

            if self.appear_then_click(ARK_GOTO_RANKING, offset=20, threshold=0.85, interval=3):
                continue

            if self.appear(ARK_CHECK, offset=10):
                location = self.appear_text('排名', threshold=0.9, interval=3)
                if location and 570 <= location[0] < 720 and 80 <= location[1] < 350:
                    self.device.click_xy(*location)
                    continue

        reward_timeout = Timer(60, count=60).start()
        while 1:
            self.device.screenshot()

            if reward_timeout.reached():
                logger.warning('Ranking reward stopped: reward collection exceeded 60 seconds')
                return False

            if self.handle_reward(interval=1):
                continue

            # 获得奖励
            if self.appear_then_click(RANKING_REWARD, threshold=30, interval=1):
                continue

            # 领取奖励
            if self.appear_then_click(RANKING_RECEIVE, offset=10, interval=1, static=False):
                continue

            # 返回
            if self.appear(RANKING_NO_REWARD, threshold=30):
                return True

    def ensure_back(self, skip_first_screenshot=True):
        click_timer = Timer(0.3)
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if click_timer.reached() and self.handle_reward(interval=1):
                click_timer.reset()
                continue

            if self.appear(SPECIAL_ARENA_CHECK, offset=(5, 5), static=False):
                break

            if click_timer.reached():
                self.device.click_xy(100, 100)
                click_timer.reset()

    def temporary(self, button, skip_first_screenshot=True):
        click_timer = Timer(0.3)
        confirm_timer = Timer(1, count=2).start()
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if click_timer.reached() and self.appear_then_click(
                button, offset=(5, 5), interval=0.3, static=False, threshold=0.9
            ):
                confirm_timer.reset()
                click_timer.reset()
                continue

            if confirm_timer.reached():
                break

    def run(self, internal_call=False):
        from module.execution.context import report, step
        # self.device.image = cv2.imread('t1.png')
        # self.appear_text('全部领取')
        self.ui_ensure(page_reward)
        step('reward.defense', '前哨基地防御收益', self.receive_reward, verified=True)
        # 友情点
        collect_social = self.config.Reward_CollectSocialPoint
        collect_special = self.config.Reward_CollectSpecialArenaPoint
        collect_ranking = self.config.Reward_CollectRanking
        if internal_call:
            collect_social = self.config.cross_get('Reward.Reward.CollectSocialPoint', default=collect_social)
            collect_special = self.config.cross_get('Reward.Reward.CollectSpecialArenaPoint', default=collect_special)
            collect_ranking = self.config.cross_get('Reward.Reward.CollectRanking', default=collect_ranking)
        if collect_social:
            # ----
            # self.ui_ensure(page_friend)
            # ----
            self.ui_ensure(page_main)
            self.temporary(MAIN_GOTO_FRIEND)
            # ----
            step('reward.social', '友情点领取检查', self.receive_social_point, verified=True)
        # pjjc奖励
        if collect_special:
            self.ui_ensure(page_arena)
            def collect_points():
                try:
                    self.receive_special_arena_point()
                    report('success', '特竞点数已检查，无剩余可领取奖励')
                except NoRewards:
                    self.ensure_back()
            step('reward.special', '特殊竞技场点数', collect_points)
        # 方舟排名奖励
        if collect_ranking:
            self.ui_ensure(page_ark)
            def collect_rank():
                if self.receive_ranking():
                    report('success', '排名奖励已检查，无剩余可领取奖励')
                else:
                    report('failed', '排名奖励入口或领取超时', retryable=True)
            step('reward.ranking', '方舟排名奖励', collect_rank)

        if not internal_call:
            self.config.task_delay(schedule=True)
