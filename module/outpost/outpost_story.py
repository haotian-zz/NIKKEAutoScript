from module.base.timer import Timer
from module.execution.context import report
from module.handler.assets import REWARD
from module.logger import logger
from module.outpost.assets import *
from module.ui.assets import SKIP
from module.ui.page import page_outpost
from module.ui.ui import UI


class NoStoryRemaining(Exception):
    pass


class NoTimesRemaining(Exception):
    pass


class OutpostStory(UI):
    # 连续点了这么多次「确认」都没进剧情，就当耐力用完
    CONFIRM_WITHOUT_STORY_LIMIT = 3

    def close_story_list(self):
        """点剧情列表底部「关闭」直到列表消失，10 秒还在就留给下一个任务的 UI 处理。"""
        timeout = Timer(10).start()
        while 1:
            self.device.screenshot()
            if not self.appear(STORY_CLOSE, offset=10) or timeout.reached():
                return
            self.appear_then_click(STORY_CLOSE, offset=10, interval=1)

    def check_remaining(self):
        # 检查次数
        while 1:
            self.device.screenshot()

            # 剧情图标
            if self.appear(STORY_OPEN, offset=10):
                # 所有剧情已看完
                if self.appear(STORY_REMAINING_0, offset=10, threshold=0.9):
                    raise NoStoryRemaining
                # 次数不够
                if self.appear(STORY_TIMES_0, offset=10, threshold=0.9):
                    raise NoTimesRemaining
                break
            else:
                continue

    def story_view(self):
        logger.hr('Story view', 2)
        self.check_remaining()

        skip = False
        confirms = 0
        while 1:
            self.device.screenshot()

            # 实录 2026-10-26：列表里看完一段后直接点下一段，耐力 0/5 时确认框的消耗变红、点确认没反应，
            # 原逻辑在「开始 → 确认 → 对话框」之间循环 7 分钟直到 GameStuck。进剧情前只在打开列表时查过次数。
            if confirms >= self.CONFIRM_WITHOUT_STORY_LIMIT:
                logger.warning(f'Story confirm clicked {confirms} times without entering a story, stamina used up')
                self.close_story_list()
                raise NoTimesRemaining

            if (
                not self.appear(STORY_DIALOG_BOX, offset=100)
                and not self.appear(STORY_CLOSE, offset=10)
                and self.appear_then_click(STORY_OPEN, offset=10, interval=1)
            ):
                logger.info('Open story list')
                continue

            # 进入剧情
            if (
                not skip
                and not self.appear(STORY_START_CONFIRM, offset=10)
                and self.appear_then_click(STORY_START, offset=10, interval=3)
            ):
                continue

            # 进入剧情确认
            if self.appear(STORY_START_CHECK, offset=10) and self.appear_then_click(
                STORY_START_CONFIRM, offset=10, interval=1
            ):
                confirms += 1
                continue

            # 任意对话框
            if not skip and self.appear_then_click(STORY_DIALOG_BOX, offset=100, click_offset=(30, -15), interval=3):
                continue

            # SKIP
            if self.appear_then_click(SKIP, offset=(150, 10), interval=1):
                skip = True
                confirms = 0
                continue

            # 关闭，回到基地主页
            if skip and self.appear_then_click(STORY_CLOSE, offset=10, interval=1):
                while 1:
                    self.device.screenshot()
                    if not self.appear(STORY_CLOSE, offset=10):
                        logger.info('Story view done')
                        self.device.click_record_clear()
                        self.device.stuck_record_clear()
                        skip = False
                        return self.story_view()
                    if self.appear_then_click(STORY_CLOSE, offset=10, interval=1):
                        continue
                continue

            # 领取
            if self.appear_then_click(REWARD, offset=10, interval=1, static=False):
                confirms = 0
                continue

    def run(self):
        logger.hr('Outpost Story')
        self.ui_ensure(page_outpost)

        try:
            self.story_view()
        except NoStoryRemaining:
            report('success', '当前可用基地剧情已处理完毕')
            logger.warning('All story allready viewed')
        except NoTimesRemaining:
            report('success', '基地剧情可用次数已用完')
            logger.warning('No times remaining for story')

        self.config.task_delay(schedule=True)
