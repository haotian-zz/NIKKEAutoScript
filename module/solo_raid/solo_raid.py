from module.execution.context import report
from module.base.timer import Timer
from module.base.watchdog import WaitGuard
from module.logger import logger
from module.simulation_room.assets import AUTO_BURST, AUTO_SHOOT, END_FIGHTING, PAUSE
from module.solo_raid.assets import *
from module.solo_raid.challenge import SoloRaidChallenge
from module.ui.assets import FIGHT_QUICKLY_CHECK, FIGHT_QUICKLY_MAX, FIGHT_QUICKLY_MIN, GOTO_MAIN, MAIN_CHECK
from module.ui.page import page_main


class NoOpportunityRemain(Exception):
    pass


class SoloRaidIsUnavailable(Exception):
    pass


class SoloRaidTeamUnavailable(Exception):
    """五组均无法入场且自动补队未解决，保留当前突击进度。"""


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

            if not self.appear(MAIN_CHECK, offset=10) and self.dismiss_raid_result():
                continue

            if confirm_timer.reached():
                logger.error('Solo raid not found')
                raise SoloRaidIsUnavailable

        if self.free_opportunity_remain:
            self.solo_raid()
        else:
            logger.warning('There are no free opportunities for normal mode')
            report('success', '普通单人突击免费次数已用完')

    def check_raid_battle_progress(self, guard):
        if self.appear(PAUSE, offset=10):
            # Long raids outlive the device watchdog; the per-battle deadline remains enforced.
            self.device.stuck_record_clear()
            guard.check()
            return True
        guard.progress()
        return False

    def solo_raid(self, skip_first_screenshot=True):
        """执行普通突击，保留已有编队，并为空队自动编排。

        空队自动编排后入场；每组入场等待最多 10 秒，五组均不可用时保留进度结束。
        实际战斗期间清除入场计时，避免把正常战斗耗时当作队伍失效。
        """
        logger.hr('Start a solo raid (Normal)')
        team_index = 0
        team_selected = False
        team_timeout = Timer(10, count=10)
        auto_attempted = set()
        repaired_at_remaining = set()
        guard = WaitGuard(self.device, '单人突击本轮战斗', timeout=1200, warn_after=180)
        transition = WaitGuard(self.device, '单人突击页面切换', timeout=45, warn_after=15)
        exhausted = Timer(3, count=3)
        battle = WaitGuard(self.device, '单人突击单场战斗', timeout=300, warn_after=180)

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            in_battle = self.check_raid_battle_progress(battle)
            guard.check()
            # 结算弹窗
            if self.appear(ENEMY_DEFEATED, offset=200) and self.appear_then_click(
                ENEMY_DEFEATED_CONFIRM, offset=200, interval=1
            ):
                break

            board = self.appear(FIGHT_HISTORY, offset=10)
            stage = self.appear(SOLO_RAID_CHECK, offset=10) and self.appear(SOLO_RAID_CHECK, threshold=30)
            if not (board or in_battle) and self.handle_raid_end_notice():
                return False
            if board or in_battle or stage:
                transition.progress()
            else:
                transition.check()
            if board and self.raid_battles_remaining() == 0:
                exhausted.start()
                if exhausted.reached():
                    report('blocked', '本次突击五次出战已用完，未确认击败 Boss', '保留剩余每日次数，请检查队伍练度')
                    return False
                continue
            exhausted.clear()

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
                team_unavailable = False
                if team_index not in auto_attempted and self.raid_team_empty():
                    auto_attempted.add(team_index)
                    if self.fill_empty_raid_team():
                        logger.info(f'Solo raid: saved team {team_index + 1} auto formation ready')
                        team_timeout.reset()
                        continue
                    team_unavailable = True
                if not team_timeout.started():
                    team_timeout.start()
                if team_unavailable or team_timeout.reached():
                    team_index += 1
                    if team_index >= 5:
                        remaining = self.raid_battles_remaining()
                        if (remaining is not None and remaining > 0 and remaining not in repaired_at_remaining
                                and self.raid_team_unavailable()):
                            repaired_at_remaining.add(remaining)
                            if self.fill_empty_raid_team(replace_unavailable=True):
                                team_index = 4
                                team_timeout.reset()
                                continue
                        raise SoloRaidTeamUnavailable
                    logger.warning(f'Solo raid team unavailable: try saved team {team_index + 1}')
                    self.device.click_xy(340 + 76 * team_index, 860)
                    team_timeout.reset()
                    continue
                if self.appear(ENTER_FIGHT, threshold=30) and self.appear_then_click(
                    ENTER_FIGHT, offset=10, interval=2
                ):
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

            if not (board or in_battle or self.appear(SOLO_RAID_CHECK, offset=10)):
                if self.dismiss_raid_result():
                    transition.progress()
                    hp = self._last_raid_result_hp
                    if hp == 0:
                        break
                    if hp is not None:
                        report('blocked', '本次突击已结算，Boss 尚未击败', '保留剩余每日次数，请检查队伍练度')
                        return False
                    continue

            # 扫荡结束
            if (
                self.appear(SOLO_RAID_CHECK, offset=10)
                and self.appear(STAGE_SEVEN, offset=(30, 30))
                and self.appear(CHALLENGE_QUICKLY_DISABLE, threshold=10)
                and not self.appear(CHALLENGE, threshold=10)
            ):
                break

        # 进入单人突击界面
        settle = WaitGuard(self.device, '单人突击结算返回', timeout=45, warn_after=15)
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            settle.check()
            if self.appear(SOLO_RAID_CHECK, offset=10):
                break
            if self.dismiss_raid_result():
                continue
            self.appear_then_click(ENEMY_DEFEATED_CONFIRM, offset=200, interval=2)

        if self.free_opportunity_remain:
            self.device.click_record_clear()
            self.device.stuck_record_clear()
            return self.solo_raid()
        else:
            logger.warning('There are no free opportunities for normal mode')
            report('success', '普通单人突击免费次数已用完')

    def enter_normal_raid(self):
        entry = WaitGuard(self.device, '单人突击任务接续', timeout=45, warn_after=15)
        while True:
            self.device.screenshot()
            entry.check()
            if self.appear(PAUSE, offset=10):
                logger.info('Solo raid: resume the active battle without restarting the game')
                return self.solo_raid()
            if self.handle_raid_end_notice():
                return False
            if self.raid_editor_visible():
                if not self.fill_empty_raid_team(editor=True):
                    raise SoloRaidTeamUnavailable
                continue
            if (self.appear(FIGHT_HISTORY, offset=10) or self.appear(SOLO_RAID_CHECK, offset=10)
                    or self.appear(MAIN_CHECK, offset=10)):
                return self.ensure_into_soloraid()
            if self.appear_then_click(END_FIGHTING, offset=30, interval=2) or self.dismiss_raid_result():
                continue
            if any(page.check_button and self.appear(page.check_button, offset=30) for page in self.ui_pages):
                self.ui_ensure(page_main)
                continue
            # 上个任务留下的活动页等没有页面模板，用左下角大厅按钮回到可识别的页面
            if self.appear_then_click(GOTO_MAIN, offset=30, interval=3):
                continue
            self.ui_additional()

    def run(self):
        from module.execution.context import report
        try:
            self.enter_normal_raid()
            # 挑战
            if self.config.SoloRaid_Challenge:
                self.ui_ensure(page_main)
                self.ensure_into_challenge()
        except SoloRaidIsUnavailable:
            report('unknown', '单人突击入口不可用', '请确认活动开放情况与入口识别', retryable=True)
        except NoOpportunityRemain:
            report('success', '单人突击当前模式免费次数已用完')
        except SoloRaidTeamUnavailable:
            report('blocked', '当前五组队伍均无法继续入场，空队或完全不可用队伍已尝试自动编排',
                   '本次突击进度已保留；请检查剩余可用角色、已出战限制和队伍练度')
            logger.warning('Solo raid stopped: saved teams cannot enter; current raid progress is preserved')

        self.config.task_delay(schedule=True)
