import re

import cv2

from module.base.button import Button
from module.base.timer import Timer
from module.base.watchdog import WaitGuard
from module.base.utils import crop
from module.conversation.assets import ANSWER_CHECK
from module.event.assets import *
from module.event.base import EventBase, EventSelectError
from module.event.challenge import CHALLENGE_QUICKLY_DISABLE
from module.execution.context import report
from module.exception import GameStuckError
from module.logger import logger
from module.ocr.ocr import Ocr
from module.simulation_room.assets import AUTO_BURST, AUTO_SHOOT, END_FIGHTING, FIGHT, FIGHT_QUICKLY
from module.tribe_tower.assets import NEXT_STAGE, OPERATION_FAILED
from module.ui.assets import (
    FIGHT_CLOSE,
    FIGHT_QUICKLY_CHECK,
    FIGHT_QUICKLY_FIGHT,
    FIGHT_QUICKLY_MAX,
)


class EventStory(EventBase):
    def STORY_PART_CHECK(self, story):
        # 只读取当前章节的模板，避免要求活动提供其他类型才会用到的资源。
        stages = {
            'story_1_normal': 'STORY_1_NORMAL',
            'story_1_hard': 'STORY_1_HARD',
            'story_2_normal': 'STORY_2_NORMAL',
            'story_2_hard': 'STORY_2_HARD',
        }
        return getattr(self.event_assets, stages[story])

    def appear_story_part(self, story, offset=30):
        button = self.STORY_PART_CHECK(story)
        if story == 'story_1_normal':
            if self.event.type == 2:
                return self.appear(button, threshold=20)
            return self.appear(button, offset=offset)
        return self.appear(button, threshold=10) and self.appear(button, offset=offset)

    def STORY_STAGE_11(self, story):
        clear = story.endswith('_clear')
        key = story[:-6] if clear else story
        return self.stage_assets(key)[-2][2 if clear else 1]

    def STORY_STAGE_12(self, story):
        clear = story.endswith('_clear')
        key = story[:-6] if clear else story
        return self.stage_assets(key)[-1][2 if clear else 1]

    def get_story_stage_pending_buttons(self, story: str) -> list:
        """
        Dynamically loads pending stage buttons from event assets.
        Searches for attributes starting with a prefix, like:
        - STORY_1_NORMAL_STAGE_PENDING
        - STORY_1_NORMAL_STAGE_PENDING_2
        """
        prefix = f'{story.upper()}_STAGE_PENDING'
        buttons = []

        for attr_name in dir(self.event_assets):
            if attr_name.startswith(prefix):
                suffix = attr_name[len(prefix) :]
                if suffix == '' or (suffix.startswith('_') and suffix[1:].isdigit()):
                    buttons.append(getattr(self.event_assets, attr_name))

        if not buttons:
            logger.warning(f"No pending stage buttons found for story '{story}' using prefix '{prefix}'")
        else:
            logger.info(f"Found {len(buttons)} pending stage buttons for story '{story}'")

        return buttons

    def find_pending_stage_by_grid(self, open_story):
        guard = WaitGuard(self.device, '活动待通关关卡搜索', timeout=120)
        finder = self.pending_finder_for(open_story)
        if finder is None:
            logger.warning(f'No pending finder configured for {open_story}')
            return 'exhausted'
        x1, y1, x2, y2 = finder['area']
        rows = finder['rows']
        columns = finder['columns']
        # 上下方向决定行顺序，左右方向决定每行内的点顺序。
        row_indices = range(rows) if finder['vertical_direction'] == 'top_to_bottom' else range(rows - 1, -1, -1)
        column_indices = (
            range(columns) if finder['horizontal_direction'] == 'left_to_right' else range(columns - 1, -1, -1)
        )
        cell_width = (x2 - x1) / columns
        cell_height = (y2 - y1) / rows
        # 配置区域均分后，按指定方向生成待点击的中心点。
        points = [
            (int(x1 + (column + 0.5) * cell_width), int(y1 + (row + 0.5) * cell_height))
            for row in row_indices
            for column in column_indices
        ]
        height, width = self.device.image.shape[:2]

        logger.info(
            f'Grid pending finder: {len(points)} points, order={finder["vertical_direction"]}/'
            f'{finder["horizontal_direction"]}'
        )
        for index, (base_x, base_y) in enumerate(points, start=1):
            if not self.appear_story_part(open_story, offset=10):
                logger.warning('Not in story list page, retry')
                return 'retry'
            # 关卡推完回到了关卡列表
            if self.appear(self.STORY_STAGE_12(open_story), offset=80, threshold=0.9) and self.appear(
                self.STORY_STAGE_12(f'{open_story}_clear'), offset=80, threshold=0.9
            ):
                logger.warning('Story push done, stage 12 cleared')
                return 'done'
            point = base_x, base_y
            if not 0 <= point[0] < width or not 0 <= point[1] < height:
                logger.warning(f'Grid pending point {index} out of screen after shift: {point}')
                return 'aborted'

            anchor_area = point[0] - 15, point[1] - 15, point[0] + 15, point[1] + 15
            anchor = Button(area=anchor_area, color=(0, 0, 0), button=anchor_area, name=f'PENDING_GRID_ANCHOR_{index}')
            anchor.load_color(self.device.image)
            anchor._match_init = True
            logger.info(f'Click grid pending point {index}/{len(points)}: {point}')
            self.device.click(anchor)

            # 点击后只接受“正常关卡且快速战斗不可用”为 pending。
            result = 'timeout'
            for _ in range(10):
                self.device.screenshot()
                guard.check()
                if self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30):
                    if not self.appear(FIGHT_QUICKLY, threshold=20):
                        logger.info(f'Grid pending stage found at {point}')
                        return 'found'
                    result = 'normal_stage'
                    break
                if self.appear(STORY_STAGE_NOT_UNLOCKED, offset=30):
                    result = 'not_unlocked'
                    break
                if self.appear(STORY_STAGE_REPEAT_FORBIDDEN, offset=30):
                    result = 'repeat_forbidden'
                    break

            if result == 'normal_stage':
                # 点到可快速战斗的普通关卡时，关闭弹窗后继续扫描。
                while 1:
                    self.device.screenshot()
                    guard.check()
                    if self.appear_then_click(FIGHT_CLOSE, offset=30, interval=1):
                        continue
                    if not self.appear(FIGHT_CLOSE, offset=30):
                        break
            elif result in ('not_unlocked', 'repeat_forbidden'):
                logger.info(f'Grid pending point {index}: {result}, wait for dialog to close')
                while 1:
                    self.device.screenshot()
                    guard.check()
                    if not self.appear(STORY_STAGE_NOT_UNLOCKED, offset=30) and not self.appear(
                        STORY_STAGE_REPEAT_FORBIDDEN, offset=30
                    ):
                        logger.info(f'Grid pending point {index}: {result}, dialog closed')
                        break

        logger.info('Grid pending finder exhausted all points')
        return 'exhausted'

    def story_modes(self):
        if self.event.type == 2:
            return ['story_1_normal', 'story_1_hard']
        return ['story_1_normal', 'story_2_normal', 'story_2_hard']

    def page_cleared(self, story):
        last = self.stage_assets(story)[-1]
        return self.appear(last[1], offset=80, threshold=0.9) and self.appear(last[2], offset=80, threshold=0.9)

    def stage_assets(self, story):
        prefix = story.upper() + '_STAGE_'
        stages = []
        for name in dir(self.event_assets):
            match = re.fullmatch(re.escape(prefix) + r'(\d+)', name)
            clear = getattr(self.event_assets, name + '_CLEAR', None)
            if match and clear is not None:
                stages.append((int(match[1]), getattr(self.event_assets, name), clear))
        if len(stages) < 2:
            raise EventSelectError('Event requires final and penultimate stage assets')
        return sorted(stages)

    def open_story_page(self, story):
        """A bounded attempt; unavailable pages fall back without abandoning the event."""
        part = int(story.split('_')[1])
        hard = story.endswith('hard')
        self.back_to_event()
        guard = WaitGuard(self.device, f'进入{story}', timeout=45, warn_after=20)
        selection = Timer(12).start()
        while True:
            self.device.screenshot()
            guard.check()
            if self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30):
                self.appear_then_click(FIGHT_CLOSE, offset=30, interval=1)
                continue
            if self.appear_story_part(story):
                logger.info(f'Event page opened: {story}')
                return True
            if self.event.type != 2:
                locked = getattr(self.event_assets, f'EVENT_GOTO_STORY_{part}_LOCKED', None)
                if locked is not None and self.appear(locked, offset=10):
                    logger.info(f'Event page locked: {story}')
                    return False
                goto = getattr(self.event_assets, f'EVENT_GOTO_STORY_{part}', None)
                if goto is not None and self.appear_then_click(goto, offset=10, interval=3):
                    selection.reset()
                    continue
            # STORY_CHECK may coincide with EVENT_CHECK for small events.
            normal = getattr(self.event_assets, f'STORY_{part}_NORMAL')
            active_normal = self.appear(normal, threshold=10) and self.appear(normal, offset=30)
            active_hard = self.appear(getattr(self.event_assets, f'STORY_{part}_HARD'), threshold=10)
            if active_normal or active_hard:
                if not hard:
                    if self.appear_then_click(normal, offset=30, interval=2):
                        continue
                else:
                    locked = getattr(self.event_assets, f'STORY_{part}_HARD_LOCKED', None)
                    if locked is not None and self.appear(locked, offset=10):
                        logger.info(f'Hard locked; fall back from {story}')
                        return False
                    if selection.reached():
                        logger.info(f'Cannot switch to {story}; fall back to previous page')
                        return False
                    button = getattr(self.event_assets, f'STORY_{part}_HARD_HIDDEN', None)
                    if button is not None and self.appear_then_click(button, threshold=10, interval=2):
                        continue
                    self.appear_then_click(getattr(self.event_assets, f'STORY_{part}_HARD'),
                                           offset=30, interval=3)
                continue
            unlock = getattr(self.event_assets, f'STORY_{part}_HARD_UNLOCK', None)
            if unlock is not None and self.appear_then_click(unlock, offset=10, interval=1):
                continue
            self.appear_then_click(getattr(self.event_assets, f'STORY_{part}_CHECK'), offset=10, interval=3)

    def story(self, skip_first_screenshot=True):
        logger.hr('EVENT PROGRESSION AND FREE TICKETS', 2)
        modes = self.story_modes()
        if not self.config.StoryStage_AutoPush:
            part = getattr(self.config, 'EventInfo_StoryPart', 'Story_1').lower()
            difficulty = getattr(self.config, 'EventInfo_StoryDifficulty', 'Hard').lower()
            target = f'{part}_{difficulty}'
            if target in modes:
                modes = modes[:modes.index(target) + 1]
        current = None
        for mode in reversed(modes):
            if self.open_story_page(mode):
                current = mode
                break
        if current is None:
            raise EventSelectError('No event story page can be entered')
        while True:
            self.device.screenshot()
            if self.config.StoryStage_AutoPush:
                result = self.find_and_push_stage(current)
                if result == 'blocked':
                    break
                if result == 'tickets':
                    report('success', f'{current} 已推进，免费票已用完；下次继续未通关关卡')
                    break
            self.device.screenshot()
            cleared = self.page_cleared(current)
            index = modes.index(current)
            if self.config.StoryStage_AutoPush and cleared and index + 1 < len(modes):
                next_mode = modes[index + 1]
                if self.open_story_page(next_mode):
                    current = next_mode
                    continue
                logger.info(f'{current} cleared; {next_mode} unavailable, use free tickets on current page')
                report('success', f'{current} 已全通，{next_mode} 尚不能进入，回当前页刷剩余票')
                if not self.open_story_page(current):
                    raise EventSelectError('Unable to return to cleared event page')
            if self.config.StoryStage_Sweep:
                self.find_and_sweep_stage(current)
            elif cleared:
                report('success', f'{current} 全关卡已通关；未启用扫荡')
            break
        self.back_to_event()

    def find_and_push_stage(self, open_story):
        guard = WaitGuard(self.device, f'{open_story}推图', timeout=300, warn_after=60)
        prepared = False
        finder = self.pending_finder_for(open_story)
        grid_mode = finder is not None and finder.get('mode') == 'grid'
        pending_buttons = [] if grid_mode else self.get_story_stage_pending_buttons(open_story)
        # 如果最后一关没有clear（与推图循环内的推完判定保持一致：两个模板同时出现才算已通关）
        if (
            not (
                self.appear(self.STORY_STAGE_12(open_story), offset=80, threshold=0.9)
                and self.appear(self.STORY_STAGE_12(f'{open_story}_clear'), offset=80, threshold=0.9)
            )
        ):
            if grid_mode:
                logger.info('Grid pending finder enabled, start pushing loop')
            else:
                logger.info('Pending stage found, start pushing loop')
            grid_retry_timer = Timer(5)
            # 判断有票和组队状态
            while 1:
                self.device.screenshot()
                guard.check()

                if self.appear(OPERATION_FAILED, offset=10):
                    report('blocked', f'{open_story} 战斗未通过', '检查角色战力和编队后再继续推图')
                    self.appear_then_click(END_FIGHTING, offset=30, interval=1)
                    return 'blocked'
                if self.appear_then_click_any([AUTO_SHOOT, AUTO_BURST], offset=10, threshold=0.9, interval=5):
                    continue

                # 打开关卡
                if grid_mode:
                    if (
                        grid_retry_timer.reached()
                        and self.appear_story_part(open_story)
                        and not self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30)
                    ):
                        result = self.find_pending_stage_by_grid(open_story)
                        if result == 'found':
                            logger.info(f'Grid pending finder {result}')
                            continue
                        if result == 'retry':
                            grid_retry_timer.reset()
                            continue
                        if result == 'done':
                            report('success', f'{open_story} 全关卡已通关')
                            return 'cleared'
                        if result in ('exhausted', 'aborted'):
                            report('unknown', '未找到可推进的关卡', '已停止网格搜索，请查看截图', retryable=True)
                            raise GameStuckError('Event pending grid exhausted')
                else:
                    was_clicked = False
                    for button in pending_buttons:
                        if self.appear_with_flip_then_click(
                            button,
                            offset=30,
                            click_offset=self.event.pending_click_offset,
                            threshold=0.9,
                            color_threshold=20,
                            interval=5,
                            static=False,
                        ):
                            logger.info(f'Click pending stage {button.name} to enter')
                            was_clicked = True
                            break
                    if was_clicked:
                        continue

                # First clears always use auto formation, even when an old event team exists.
                if self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30) and not prepared:
                    if self.appear(FIGHT_QUICKLY, threshold=20):
                        self.appear_then_click(FIGHT_CLOSE, offset=30, interval=1)
                        continue
                    if self.config.StoryStage_AutoFormation:
                        self.open_team()
                        self.team_up(bonus=False)
                    prepared = True
                    continue

                # 没票退出
                if (
                    self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30)
                    and not self.appear(FIGHT, threshold=20)
                    and self.appear(FIGHT_CLOSE, offset=10)
                ):
                    # 重复确认
                    confirm_timer = Timer(2, count=3).start()
                    no_ticket = False
                    while 1:
                        self.device.screenshot()
                        guard.check()
                        # 中途出现 FIGHT 或弹窗已关，说明不是没票，回外层继续遍历
                        if not self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30) or self.appear(
                            FIGHT, threshold=20
                        ):
                            break
                        if confirm_timer.reached():
                            no_ticket = True
                            logger.warning('Story push done, no ticket')
                            break
                    if no_ticket:
                        # 关闭弹窗，点击未生效则重试
                        while 1:
                            self.device.screenshot()
                            guard.check()
                            if self.appear_then_click(FIGHT_CLOSE, offset=10, interval=1):
                                continue
                            if not self.appear(FIGHT_CLOSE, offset=10):
                                break
                        # 没票直接退出
                        if self.event_tickets(open_story) == 0:
                            return 'tickets'
                        report('unknown', '入场按钮不可用，未确认免费票为零', retryable=True)
                        raise GameStuckError('Event entry unavailable; ticket count not confirmed zero')
                # 进入战斗
                if (
                    self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30)
                    and not self.appear(FIGHT_QUICKLY, threshold=10)
                    and self.appear_then_click(FIGHT, threshold=20, interval=5)
                ):
                    continue

                # 剧情选项
                if self.appear_then_click(ANSWER_CHECK, offset=100, interval=3):
                    continue
                # 跳过剧情
                if self.appear_then_click(self.event_assets.SKIP, offset=(150, 10), interval=1):
                    continue
                # 剧情推完提示
                if self.appear_then_click(STAGE_COMPLETE_CONFIRM, offset=10, interval=1):
                    continue

                # 下一关卡
                if self.appear_then_click(NEXT_STAGE, offset=(100, 30), interval=1):
                    self.device.stuck_record_clear()
                    self.device.click_record_clear()
                    prepared = False
                    guard.progress()
                    continue

                # 点击区域跳转
                if self.appear_then_click(FIELD_CHANGE, offset=30, interval=1):
                    continue

                # 回到活动主页
                if self.appear(self.event_assets.EVENT_CHECK, offset=(30, 30)):
                    logger.info('Returned to event page, restart story loop')
                    raise GameStuckError('Event returned home before confirming page completion')

                # 战斗结束，但是没有找到下一关卡
                if self.appear_then_click(END_FIGHTING, offset=30, interval=1):
                    logger.info('Story battle ended')
                    prepared = False
                    guard.progress()
                    continue

                # 关卡推完回到了关卡列表
                if self.appear(self.STORY_STAGE_12(open_story), offset=80, threshold=0.9) and self.appear(
                    self.STORY_STAGE_12(f'{open_story}_clear'), offset=80, threshold=0.9
                ):
                    logger.warning('Story push done, stage 12 cleared')
                    report('success', f'{open_story} 全关卡已通关')
                    return 'cleared'
        else:
            logger.info('Final stage already cleared')
            report('success', f'{open_story} 全关卡已通关')
            return 'cleared'


    def event_tickets(self, open_story=None):
        def read_counter():
            # The counter is absent on some story maps and hidden by stage panels.
            # Read each location independently so unrelated OCR cannot form a fake fraction.
            for area in ((350, 10, 460, 85), (520, 305, 705, 350)):
                result = Ocr(area, name='EVENT_FREE_TICKETS',
                             model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)
                match = re.fullmatch(r'\s*(\d+)\s*[/／]\s*5\s*', result['text'])
                if not match and result['text'].strip():
                    # Tiny white slashes may disappear in the unprocessed RGB screenshot.
                    result = Ocr(area, name='EVENT_FREE_TICKETS_WHITE',
                                 model_type=self.config.Optimization_OcrModelType,
                                 text_color=(255, 255, 255),
                                 text_color_tolerance=(179, 100, 100)).ocr(self.device.image)
                    match = re.fullmatch(r'\s*(\d+)\s*[/／]\s*5\s*', result['text'])
                if match:
                    return int(match[1])
            return None

        tickets = read_counter()
        if tickets is None and open_story is not None:
            self.back_to_event()
            self.device.screenshot()
            tickets = read_counter()
            if tickets != 0 and not self.open_story_page(open_story):
                raise EventSelectError('Unable to return to story after checking event tickets')
        return tickets

    def open_team(self):
        guard = WaitGuard(self.device, '活动打开编队', timeout=30, warn_after=15)
        while True:
            self.device.screenshot()
            guard.check()
            if self.appear(SAVE_TEAM, offset=30) and self.appear(TEAM_NIKKE_AUTO, offset=30):
                return
            if self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30):
                self.device.click_xy(620, 975)
                self.device.sleep(0.5)

    def team_slot_empty(self, index):
        TEAM_NIKKE_NOT_SELECT.ensure_template()
        x = 110 + 125 * index
        image = crop(self.device.image, (x - 25, 338, x + 25, 390))
        result = cv2.matchTemplate(image, TEAM_NIKKE_NOT_SELECT.image, cv2.TM_CCOEFF_NORMED)
        return cv2.minMaxLoc(result)[1] >= 0.85

    def sort_team(self, bonus):
        label = '加成奖励' if bonus else '战斗力'
        self.device.click_xy(550, 525)
        self.device.sleep(0.3)
        self.device.screenshot()
        if not self.appear_text_then_click(label, threshold=0.9):
            raise GameStuckError(f'Event team sort option unavailable: {label}')
        self.device.click_xy(165, 580)
        self.device.sleep(0.3)
        self.device.screenshot()
        # Downward arrow means highest first; enforce it instead of inheriting a user's previous direction.
        arrow = self.device.image[518:536, 641:662]
        gray = cv2.cvtColor(arrow, cv2.COLOR_RGB2GRAY)
        bright = gray > 150
        upper = int(bright[:9].sum())
        lower = int(bright[9:].sum())
        if lower > upper:
            self.device.click_xy(650, 525)
            self.device.sleep(0.3)

    def team_up(self, bonus=False):
        guard = WaitGuard(self.device, '活动加成编队' if bonus else '活动最强编队', timeout=90)
        self.device.screenshot()
        if not self.appear(SAVE_TEAM, offset=30):
            raise GameStuckError('Event formation page not confirmed')
        # This reset affects only the event squad. Auto formation on a full team need not replace it.
        self.device.click_xy(309, 1205)
        self.device.sleep(0.3)
        self.device.screenshot()
        if not all(self.team_slot_empty(i) for i in range(5)):
            raise GameStuckError('Event team reset not confirmed')
        if not bonus:
            self.device.click(TEAM_NIKKE_AUTO)
            self.device.sleep(0.5)
        else:
            self.sort_team(bonus=True)
            use_bonus = True
            for index in range(5):
                guard.check()
                self.device.screenshot()
                # Selected cards are pinned before the remaining sorted roster.
                column, row = index % 4, index // 4
                x = (110, 277, 443, 610)[column]
                y = 645 + row * 320
                if use_bonus:
                    result = Ocr((x - 65, y - 20, x + 65, y + 20), name='EVENT_TEAM_BONUS',
                                 model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)
                    match = re.search(r'(\d+)\s*[%％]', result['text'])
                    if not match or int(match[1]) == 0:
                        use_bonus = False
                        self.sort_team(bonus=False)
                        self.device.screenshot()
                    else:
                        logger.info(f'Event bonus slot {index + 1}: +{match[1]}%')
                self.device.click_xy(x, y + 90)
                self.device.sleep(0.3)
                self.device.screenshot()
                if self.team_slot_empty(index):
                    raise GameStuckError(f'Event team slot {index + 1} selection not confirmed')
        while True:
            self.device.screenshot()
            guard.check()
            if self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30):
                logger.info('Event team saved: bonus priority' if bonus else 'Event strongest team saved')
                return
            if any(self.team_slot_empty(i) for i in range(5)):
                raise GameStuckError('Event team is not full')
            self.appear_then_click(SAVE_TEAM, offset=30, interval=2)

    def find_and_sweep_stage(self, open_story):
        guard = WaitGuard(self.device, f'{open_story}倒数第二关扫荡', timeout=120)
        self.device.screenshot()
        if self.event_tickets(open_story) == 0:
            report('success', f'{open_story} 免费活动票已用完')
            return
        self.ensure_sroll_to_bottom(x1=(680, 1000), x2=(680, 480), count=3, delay=0.4)
        self.device.screenshot()
        stage, button, clear = self.stage_assets(open_story)[-2]
        if not self.appear(clear, offset=80, threshold=0.9):
            report('blocked', '倒数第二关尚未通关，保留票用于后续推图')
            return
        logger.info(f'Event repeat target: {open_story}, penultimate asset stage {stage}')
        prepared, submitted, completed = False, False, False
        max_clicks = 0
        while True:
            self.device.screenshot()
            guard.check()
            if self.appear(END_FIGHTING, offset=30):
                completed = True
                self.appear_then_click(END_FIGHTING, offset=30, interval=1)
                continue
            if self.appear(self.event_assets.STORY_STAGE_CHECK, offset=30):
                if self.appear(CHALLENGE_QUICKLY_DISABLE, threshold=10):
                    self.appear_then_click(FIGHT_CLOSE, offset=30, interval=1)
                    if self.event_tickets(open_story) == 0:
                        report('success', f'{open_story} 免费活动票已用完，刷票选择倒数第二关')
                    else:
                        report('unknown', '快速战斗不可用，未确认票数为零', retryable=True)
                    return
                if not prepared:
                    label = Ocr((30, 680, 350, 750), name='EVENT_REPEAT_STAGE',
                                model_type=self.config.Optimization_OcrModelType).ocr(self.device.image)['text']
                    logger.info(f'Event repeat stage verified on panel: {label}')
                    self.open_team()
                    self.team_up(bonus=True)
                    prepared = True
                    continue
                if self.appear_then_click(FIGHT_QUICKLY, threshold=20, interval=2):
                    continue
            if self.appear(FIGHT_QUICKLY_CHECK, offset=10):
                # 上游 f636feba：固定点满 3 次 MAX 再战斗
                if max_clicks < 3 and self.appear_then_click(FIGHT_QUICKLY_MAX, offset=30,
                                                           threshold=0.99, interval=0.5):
                    max_clicks += 1
                    self.device.sleep(0.4)
                    continue
                if max_clicks >= 3 and self.appear_then_click(FIGHT_QUICKLY_FIGHT, threshold=20, interval=2):
                    submitted = True
                    continue
            if self.appear_story_part(open_story) and not self.appear(
                self.event_assets.STORY_STAGE_CHECK, offset=30
            ):
                if submitted and completed:
                    tickets = self.event_tickets(open_story)
                    if tickets == 0:
                        report('success', f'{open_story} 倒数第二关加成编队扫荡完成，免费票为零')
                        return
                    if tickets is None:
                        report('unknown', '扫荡已结算，剩余票数未识别', retryable=True)
                        return
                    logger.info(f'Event sweep settled, {tickets} tickets left; continue same stage')
                    submitted, completed, max_clicks = False, False, 0
                    guard.progress()
                self.appear_then_click(button, offset=80, threshold=0.9, interval=3)
