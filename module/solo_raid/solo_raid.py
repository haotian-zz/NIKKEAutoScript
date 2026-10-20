from module.execution.context import report
from module.base.timer import Timer
from module.logger import logger
from module.simulation_room.assets import AUTO_BURST, AUTO_SHOOT, END_FIGHTING, PAUSE
from module.solo_raid.assets import *
from module.solo_raid.challenge import SoloRaidChallenge
from module.ui.assets import FIGHT_QUICKLY_CHECK, FIGHT_QUICKLY_MAX, FIGHT_QUICKLY_MIN, MAIN_CHECK
from module.ui.page import page_main


class NoOpportunityRemain(Exception):
    pass


class SoloRaidIsUnavailable(Exception):
    pass


class SoloRaidTeamUnavailable(Exception):
    """已保存的五组编队均无法入场，保留当前突击进度供手动补配队伍。"""


class SoloRaid(SoloRaidChallenge):
    def ensure_into_soloraid(self, skip_first_screenshot=True):
        """进入单人突击"""
        logger.hr('SOLO RAID START')
        click_timer = Timer(0.3)
        confirm_timer = Timer(15, count=10).start()

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            # 结算
            if click_timer.reached() and self.appear_then_click(RAID_RESULT, offset=10, interval=3, static=False):
                logger.info('Solo raid has end')
                raise SoloRaidIsUnavailable

            if (
                click_timer.reached()
                and self.appear(MAIN_CHECK, offset=10)
                and self.appear_then_click(SOLO_RAID, offset=10, interval=3, static=False)
            ):
                logger.info('Enter solo raid')
                continue

            # 选择第七关
            if (
                click_timer.reached()
                and self.appear(SOLO_RAID_CHECK, offset=(10, 10))
                and self.appear(STAGE_CHALLENGE, offset=(30, 30))
                and self.appear_then_click(STAGE_SEVEN_SWITCH, offset=10, interval=1)
            ):
                click_timer.reset()
                continue

            if self.appear(FIGHT_HISTORY, offset=10):
                logger.info('Resume the unfinished normal solo raid')
                self.solo_raid()
                return

            if self.appear(SOLO_RAID_CHECK, offset=(10, 10)) and not self.appear(STAGE_CHALLENGE, offset=(30, 30)):
                break

            if confirm_timer.reached():
                logger.error('Solo raid not found')
                raise SoloRaidIsUnavailable

        if self.free_opportunity_remain:
            self.solo_raid()
        else:
            logger.warning('There are no free opportunities for normal mode')
            report('success', '普通单人突击免费次数已用完')

    def solo_raid(self, skip_first_screenshot=True):
        """执行普通突击，入场失败时依次尝试已有编队，不修改角色组成。

        每组编队等待最多 10 秒；五组均不可用时保留当前战斗进度并结束本任务。
        实际战斗期间清除入场计时，避免把正常战斗耗时当作队伍失效。
        """
        logger.hr('Start a solo raid (Normal)')
        team_index = 0
        team_selected = False
        team_timeout = Timer(10, count=10)

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            # 第七关扫荡
            if (
                self.appear(SOLO_RAID_CHECK, offset=10)
                and self.appear(STAGE_SEVEN, offset=(30, 30))
                and self.appear_then_click(CHALLENGE_QUICKLY_ENABLE, threshold=10, interval=1)
            ):
                continue

            # 扫荡票max
            if self.appear(FIGHT_QUICKLY_CHECK, offset=10) and self.appear_then_click(
                FIGHT_QUICKLY_MAX, offset=30, threshold=0.99, interval=1
            ):
                continue

            # 扫荡确定
            if (
                self.appear(FIGHT_QUICKLY_CHECK, offset=10)
                and self.appear(FIGHT_QUICKLY_MIN, offset=30, threshold=0.99)
                and self.appear_then_click(CHALLENGE_QUICKLY_CONFIRM, offset=10, interval=1)
            ):
                continue

            # 挑战
            if (
                self.appear(SOLO_RAID_CHECK, offset=10)
                and self.appear(CHALLENGE_QUICKLY_DISABLE, threshold=10)
                and self.appear_then_click(CHALLENGE, threshold=10, interval=2)
            ):
                continue

            # 挑战确认
            if self.appear(CHALLENGE_CONFIRM_CHECK, offset=10) and self.appear_then_click(
                CHALLENGE_CONFIRM, offset=10, interval=1
            ):
                continue

            # 开始战斗
            if self.appear(FIGHT_HISTORY, offset=10):
                if not team_selected:
                    self.device.click_xy(340, 860)
                    team_selected = True
                    team_timeout.start()
                    continue
                if not team_timeout.started():
                    team_timeout.start()
                if team_timeout.reached():
                    team_index += 1
                    if team_index >= 5:
                        raise SoloRaidTeamUnavailable
                    logger.warning(f'Solo raid team unavailable: try saved team {team_index + 1}')
                    self.device.click_xy(340 + 76 * team_index, 860)
                    team_timeout.reset()
                    continue
                if self.appear_then_click(ENTER_FIGHT, offset=10, interval=1):
                    continue
            else:
                team_timeout.clear()

            # 自动射击和爆裂
            if self.appear_then_click(AUTO_SHOOT, offset=10, threshold=0.9, interval=5):
                continue
            if self.appear_then_click(AUTO_BURST, offset=10, threshold=0.9, interval=5):
                continue
            # 红圈
            if self.config.Optimization_AutoRedCircle and self.appear(PAUSE, offset=(5, 5)):
                if self.handle_red_circles():
                    continue

            # 结束
            if self.appear(END_FIGHTING, offset=30):
                while 1:
                    self.device.screenshot()
                    if not self.appear(END_FIGHTING, offset=30):
                        break
                    if self.appear_then_click(END_FIGHTING, offset=30, interval=1):
                        continue
                continue

            # 结算弹窗
            if self.appear(ENEMY_DEFEATED, offset=200) and self.appear_then_click(
                ENEMY_DEFEATED_CONFIRM, offset=200, interval=1
            ):
                break

            # 扫荡结束
            if (
                self.appear(SOLO_RAID_CHECK, offset=10)
                and self.appear(STAGE_SEVEN, offset=(30, 30))
                and self.appear(CHALLENGE_QUICKLY_DISABLE, threshold=10)
                and not self.appear(CHALLENGE, threshold=10)
            ):
                break

        # 进入单人突击界面
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.appear(SOLO_RAID_CHECK, offset=10):
                break

        if self.free_opportunity_remain:
            self.device.click_record_clear()
            self.device.stuck_record_clear()
            return self.solo_raid()
        else:
            logger.warning('There are no free opportunities for normal mode')
            report('success', '普通单人突击免费次数已用完')

    def run(self):
        from module.execution.context import report
        try:
            self.ui_ensure(page_main)
            # 普通
            self.ensure_into_soloraid()
            # 挑战
            if self.config.SoloRaid_Challenge:
                self.ui_ensure(page_main)
                self.ensure_into_challenge()
        except SoloRaidIsUnavailable:
            report('unknown', '单人突击入口不可用', '请确认活动开放情况与入口识别', retryable=True)
        except NoOpportunityRemain:
            report('success', '单人突击当前模式免费次数已用完')
        except SoloRaidTeamUnavailable:
            report('blocked', '保存的队伍均无法继续入场', '本次突击进度已保留，请先配置其他可用队伍')
            logger.warning('Solo raid stopped: saved teams cannot enter; current raid progress is preserved')

        self.config.task_delay(schedule=True)
