from functools import cached_property
import re

import cv2

from module.base.watchdog import WaitGuard
from module.exception import GameStuckError
from module.execution.context import report
from module.event.assets import SAVE_TEAM, TEAM_NIKKE_AUTO, TEAM_NIKKE_NOT_SELECT

from module.base.timer import Timer
from module.logger import logger
from module.ocr.ocr import Digit, Ocr
from module.simulation_room.assets import AUTO_BURST, AUTO_SHOOT, END_FIGHTING, PAUSE
from module.solo_raid.assets import *
from module.ui.assets import MAIN_CHECK
from module.ui.ui import UI


class SoloRaidChallenge(UI):
    def raid_team_empty(self, editor=False):
        if self.device.image.shape[:2] != (1280, 720):
            return False
        TEAM_NIKKE_NOT_SELECT.ensure_template()
        for index in range(5):
            x = 110 + 125 * index if editor else 107 + 128 * index
            y = 363 if editor else 968
            roi = self.device.image[y - 35:y + 40, x - 35:x + 35]
            match = cv2.matchTemplate(roi, TEAM_NIKKE_NOT_SELECT.image, cv2.TM_CCOEFF_NORMED)
            if cv2.minMaxLoc(match)[1] < 0.85:
                return False
        return True

    def raid_team_unavailable(self, editor=False):
        area = (40, 338, 685, 395) if editor else (45, 940, 685, 995)
        text = Ocr(area, name='SOLO_RAID_UNAVAILABLE_MEMBERS',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        return re.sub(r'\s+', '', text).count('无法选择') == 5

    def active_raid_team(self, editor=False):
        active = []
        for index in range(5):
            x = 338 + 80 * index if editor else 340 + 76 * index
            y = 190 if editor else 860
            red, green, blue = cv2.mean(self.device.image[y - 18:y + 18, x - 18:x + 18])[:3]
            if green > 90 and blue > 120 and green - red > 30 and blue - red > 60:
                active.append(index)
        return active[0] if len(active) == 1 else None

    def raid_editor_visible(self):
        if not (self.appear(SAVE_TEAM, offset=30) and self.appear(TEAM_NIKKE_AUTO, offset=30)):
            return False
        text = Ocr((25, 150, 280, 225), name='SOLO_RAID_TEAM_EDITOR',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        return 'SOLO' in text.upper() and 'RAID' in text.upper()

    def fill_empty_raid_team(self, editor=False, replace_unavailable=False):
        """Use the game's formation rules for an empty raid preset; retain existing squads."""
        guard = WaitGuard(self.device, '单人突击自动编队', timeout=45, warn_after=15)
        unavailable_preset = None
        if not editor:
            if not self.appear(FIGHT_HISTORY, offset=10):
                return False
            if not self.raid_team_empty():
                if not (replace_unavailable and self.raid_team_unavailable()):
                    return False
                unavailable_preset = self.active_raid_team()
                if unavailable_preset is None:
                    raise GameStuckError('Solo raid selected preset could not be verified')
            self.device.click_xy(107, 968)
        clicked_auto = False
        save_requested = False
        no_eligible = False
        auto_wait = Timer(3, count=3)
        while True:
            self.device.screenshot()
            guard.check()
            if self.appear(FIGHT_HISTORY, offset=10):
                if save_requested:
                    return not self.raid_team_empty()
                if no_eligible:
                    return False
                self.device.click_xy(107, 968)
                continue
            if not self.raid_editor_visible():
                continue
            if no_eligible:
                self.device.click_xy(80, 1205)
                self.device.sleep(0.5)
                continue
            if not clicked_auto:
                if not self.raid_team_empty(editor=True):
                    # The editor omits unavailable labels; carry the verified board state across the same tab.
                    rebuild = unavailable_preset is not None or (
                        replace_unavailable and self.raid_team_unavailable(editor=True)
                    )
                    if rebuild:
                        if unavailable_preset is not None and self.active_raid_team(editor=True) != unavailable_preset:
                            raise GameStuckError('Solo raid preset changed before unavailable formation reset')
                        logger.info('Solo raid: rebuild a preset whose five members are unavailable')
                        self.device.click_xy(310, 1205)
                        reset = Timer(3, count=3).start()
                        while True:
                            self.device.sleep(0.2)
                            self.device.screenshot()
                            guard.check()
                            if self.raid_team_empty(editor=True):
                                break
                            if reset.reached():
                                raise GameStuckError('Solo raid unavailable formation reset was not confirmed')
                    else:
                        logger.info('Solo raid: retain existing formation')
                        save_requested = True
                        self.appear_then_click(SAVE_TEAM, offset=30, interval=2)
                        continue
                logger.info('Solo raid: auto arrange unused eligible characters for an empty preset')
                self.device.click(TEAM_NIKKE_AUTO)
                clicked_auto = True
                auto_wait.start()
                continue
            if self.raid_team_empty(editor=True):
                if auto_wait.reached():
                    logger.warning('Solo raid: auto formation found no eligible characters')
                    self.device.click_xy(80, 1205)
                    self.device.sleep(0.5)
                    no_eligible = True
                continue
            save_requested = True
            self.appear_then_click(SAVE_TEAM, offset=30, interval=2)

    def handle_raid_end_notice(self):
        text = Ocr((180, 500, 570, 650), name='SOLO_RAID_END_NOTICE',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        text = re.sub(r'\s+', '', text)
        if '挑战结束' not in text or not any(label in text for label in ('记录已被重置', '纪录已被重置')):
            return False
        label = Ocr((245, 760, 480, 835), name='SOLO_RAID_END_CONFIRM',
                    model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        if '确认' not in label:
            return False
        logger.info('Solo raid: acknowledge the ended attempt and preserve remaining daily opportunities')
        self.device.click_xy(360, 800)
        self.device.sleep(0.5)
        report('blocked', '本轮突击已结束，游戏提示记录已重置',
               '未确认通关，剩余每日次数已保留；请核对关卡进度并调整队伍后再挑战')
        return True

    def raid_battles_remaining(self):
        text = Ocr((430, 1095, 695, 1145), name='SOLO_RAID_BATTLES_REMAINING',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        match = re.search(r'(?<!\d)([0-5])\s*[/／]\s*5(?!\d)', text)
        return int(match[1]) if match else None

    def dismiss_raid_result(self):
        text = Ocr((180, 1060, 610, 1180), name='SOLO_RAID_RESULT_NEXT',
                   model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
        if '点击任意' in text.replace(' ', ''):
            header = Ocr((175, 650, 700, 815), name='SOLO_RAID_RESULT_HP',
                         model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
            match = re.search(r'([\d,]+)\s*[/／]\s*([\d,]+)', header)
            self._last_raid_result_hp = None
            if 'HPREMAINING' in re.sub(r'\s+', '', header.upper()) and match:
                remaining, total = (int(value.replace(',', '')) for value in match.groups())
                if remaining <= total:
                    self._last_raid_result_hp = remaining
            logger.info('Solo raid: close the completed battle result')
            self.device.click_xy(360, 1130)
            self.device.sleep(0.5)
            return True
        return False

    @cached_property
    def teams(self):
        return [RAID_TEAM_1, RAID_TEAM_2, RAID_TEAM_3, RAID_TEAM_4, RAID_TEAM_5]

    @property
    def free_remain(self) -> int:
        model_type = self.config.Optimization_OcrModelType
        FREE_REMAIN = Digit(
            [FREE_OPPORTUNITY_CHECK.area],
            name='FREE_REMAIN',
            model_type=model_type,
            lang='ch',
        )
        return int(FREE_REMAIN.ocr(self.device.image)['text'])

    @property
    def free_opportunity_remain(self) -> bool:
        # result = self.appear(FREE_OPPORTUNITY_CHECK, offset=10, threshold=0.8)
        if self.free_remain:
            logger.info(f'[Free opportunities remain] {self.free_remain}')
        return self.free_remain

    @property
    def challenge_damage_is_zero(self) -> bool:
        """
        判断挑战模式当前伤害是否为0
        """
        if self.appear(CHALLENGE_DAMAGE_ZERO, offset=(30, 30)):
            return True
        else:
            return False

    def ensure_into_challenge(self, skip_first_screenshot=True):
        """检查并进入挑战模式"""
        logger.hr('CHALLENGE MODE CHECK')

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.appear(MAIN_CHECK, offset=10) and self.appear_then_click(
                SOLO_RAID, offset=10, interval=3, static=False
            ):
                logger.info('Enter solo raid')
                continue

            # 检查挑战
            if self.appear(SOLO_RAID_CHECK, offset=(10, 10)):
                if self.appear(STAGE_CHALLENGE, offset=(30, 30)):
                    # 挑战界面
                    break
                else:
                    # 挑战未开启
                    logger.warning('Challenge mode not reached')
                    return

        # 检查伤害是否为0
        if not self.challenge_damage_is_zero:
            logger.info('Challenge damage is already recorded. Skip challenge mode.')
            return

        # 检查挑战次数
        if self.free_opportunity_remain:
            self.device.click_record_clear()
            self.device.stuck_record_clear()
            self.challenge_raid()
        else:
            logger.warning('There are no free opportunities for challenge mode')

    def challenge_raid(self, skip_first_screenshot=True):
        """挑战模式战斗执行"""
        logger.hr('Start a challenge raid')
        team_change_timer = Timer(1).start()
        auto_attempted = set()

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            # 点击挑战
            if self.appear(SOLO_RAID_CHECK, offset=10) and self.appear_then_click(CHALLENGE, threshold=10, interval=2):
                continue

            # 挑战确认
            if self.appear(CHALLENGE_CONFIRM_CHECK, offset=10) and self.appear_then_click(
                CHALLENGE_CONFIRM, offset=10, interval=1
            ):
                continue

            # 选择队伍
            if self.appear(FIGHT_HISTORY, offset=10):
                if self.raid_team_empty():
                    # A failed empty preset must not trigger formation forever.
                    selected = next((i for i, team in enumerate(self.teams)
                                     if not self.appear(team, threshold=10)), -1)
                    if selected not in auto_attempted:
                        auto_attempted.add(selected)
                        if self.fill_empty_raid_team():
                            team_change_timer.reset()
                            continue
                if self.appear(ENTER_FIGHT, offset=10) and self.appear_then_click(
                    ENTER_FIGHT, threshold=10, interval=1
                ):
                    team_change_timer.reset()
                    continue

                if team_change_timer.reached():
                    current_team = -1
                    for i, team in enumerate(self.teams):
                        if not self.appear(team, threshold=10):
                            current_team = i
                            break

                    if current_team != -1:
                        if current_team < 4:
                            logger.info(f'Team {current_team + 1} is not valid, switch to Team {current_team + 2}')
                            self.device.click(self.teams[current_team + 1])
                            self.device.sleep(0.5)
                            team_change_timer.reset()
                            continue
                        else:
                            if 4 not in auto_attempted and self.raid_team_unavailable():
                                auto_attempted.add(4)
                                if self.fill_empty_raid_team(replace_unavailable=True):
                                    team_change_timer.reset()
                                    continue
                            logger.warning('No valid team found')
                            break
            else:
                team_change_timer.reset()

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

            # 回到队伍选择界面
            if self.appear(FIGHT_HISTORY, offset=10) and self.appear(RAID_TEAM_2, threshold=10):
                logger.info('Challenge raid end team one')
                continue

            # 结算弹窗
            if self.appear(ENEMY_DEFEATED, offset=150) and self.appear(ENEMY_DEFEATED_CONFIRM, offset=150):
                logger.info('Challenge raid end team all')

                while 1:
                    self.device.screenshot()
                    if self.appear(SOLO_RAID_CHECK, offset=30):
                        break
                    if self.appear_then_click(ENEMY_DEFEATED_CONFIRM, offset=150, interval=1):
                        continue
                break

        # 如果次数仍大于0且伤害依旧为0，继续挑战
        if self.free_opportunity_remain:
            if not self.challenge_damage_is_zero:
                logger.info('Challenge damage is recorded. Challenge mode complete.')
            else:
                logger.warning('Challenge mode complete. But damage is not recorded.')
