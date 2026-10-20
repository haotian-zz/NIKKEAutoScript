from module.base.timer import Timer
from module.coop.coop import CoopIsUnavailable
from module.event.assets import TEMPLATE_SHOP_GEM
from module.event.base import ChallengeNotFoundError, EventSelectError, EventUnavailableError
from module.event.challenge import EventChallenge
from module.event.coop import EventCoop
from module.event.extend import EventExtend
from module.event.game import EventGame
from module.event.login import EventLogin
from module.event.reward import EventReward
from module.event.shop import EventShop
from module.event.story import EventStory
from module.exception import GameStuckError
from module.logger import logger
from module.ui.assets import EVENT_SWITCH, MAIN_CHECK, SKIP
from module.ui.page import page_main


class Event(
    EventExtend,
    EventLogin,
    EventChallenge,
    EventReward,
    EventStory,
    EventCoop,
    EventShop,
    EventGame,
):
    def ensure_into_event(self, skip_first_screenshot=True):
        logger.hr('OPEN EVENT STORY', 2)
        click_timer = Timer(0.3)
        confirm_timer = Timer(30, count=20).start()
        event_timer = Timer(3, count=5)
        skip_click_timer = Timer(5)

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            # 检查活动主页
            if self.appear(self.event_assets.EVENT_CHECK, offset=(30, 30)):
                if not event_timer.started():
                    event_timer.start()
                if event_timer.reached():
                    break
            else:
                event_timer.clear()

            if confirm_timer.reached():
                from module.base.watchdog import save_wait_image
                from module.execution.context import report
                save_wait_image(self.device, 'event_entry_timeout')
                report('failed', '活动入口未能打开', '30 秒内未进入活动，不能据此判定活动已结束', retryable=True)
                raise GameStuckError('Event entry did not open within 30 seconds')

            # 切换活动
            if (
                click_timer.reached()
                and self.appear(MAIN_CHECK, offset=10)
                and self.appear_then_click(EVENT_SWITCH, offset=10, interval=3)
            ):
                click_timer.reset()
                continue

            # 在主页点击故事图标
            if (
                click_timer.reached()
                and self.appear(MAIN_CHECK, offset=10)
                and self.appear_then_click(self.event_assets.MAIN_GOTO_EVENT, offset=(50, 50), interval=5)
            ):
                click_timer.reset()
                logger.info('Open event story')
                continue

            # 跳过剧情
            if click_timer.reached() and self.appear_then_click(SKIP, offset=(150, 10), interval=2):
                click_timer.reset()
                confirm_timer.reset()
                continue

            # 跳过加载和第一次进入时的continue
            if (
                skip_click_timer.reached()
                and not self.appear(self.event_assets.EVENT_CHECK, offset=(30, 30))
                and not self.appear(MAIN_CHECK, offset=10)
            ):
                self.device.click_xy(10, 10)
                skip_click_timer.reset()

    def run(self):
        from module.execution.context import plan_steps, report, step
        items = [('event.reward', '活动奖励'), ('event.extend', '活动额外内容')]
        if self.config.Event_LoginStamp:
            items.append(('event.login', '活动签到'))
        if self.config.Event_Challenge:
            items.append(('event.challenge', '活动挑战'))
        if self.config.StoryStage_HardTicket and self.config.EventInfo_StoryDifficulty == 'Hard':
            items.append(('event.tickets', '活动票购买'))
        if self.config.StoryStage_AutoPush or self.config.StoryStage_Sweep:
            items.append(('event.story', '活动推图 / 剩余票扫荡'))
        if self.config.Event_Game:
            items.append(('event.game', '活动小游戏'))
        if self.config.Event_Shop:
            items.append(('event.shop', '活动商店'))
        plan_steps(items)
        # self.team_up()
        # image = cv2.imread('1.png')
        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=image)
        # self.device.image  = image
        # self.appear_text('8')

        # 是否需要重新执行
        # coop_reschedule = False

        try:
            self.ui_ensure(page_main)
            _ = self.event

            self.ensure_into_event()
            if self.config.Event_LoginStamp:
                step('event.login', '活动签到', self.login_stamp, verified=True)
            if self.config.Event_Challenge:
                step('event.challenge', '活动挑战', self.challenge, verified=True)
            # 提前买票
            if self.config.StoryStage_HardTicket and self.config.EventInfo_StoryDifficulty == 'Hard':
                step('event.tickets', '活动票购买', self.shop, TEMPLATE_SHOP_GEM)
                self.back_to_event()
            if self.config.StoryStage_AutoPush or self.config.StoryStage_Sweep:
                step('event.story', '活动推图 / 剩余票扫荡', self.story)
            # if self.config.Event_Coop:
            #     coop_reschedule = self.coop()
            if self.config.Event_Game:
                step('event.game', '活动小游戏', self.game)

            step('event.reward', '活动奖励', self.reward, verified=True)

            step('event.extend', '活动额外内容', self.event_extend)

            if self.config.Event_Shop:
                step('event.shop', '活动商店', self.shop)

        except EventSelectError:
            report('blocked', '活动关卡配置与当前进度不匹配', '请检查关卡和难度，完成必要的首次通关')
            logger.error('The event stage/difficulty select wrong')
        except EventUnavailableError:
            report('skipped', '当前配置的活动已结束或不可用')
            logger.error('The event is no longer available')
        except ChallengeNotFoundError:
            report('unknown', '活动挑战入口或关卡未识别', '请确认挑战开放情况与编队', retryable=True)
            logger.error('Challenge stage not found')
        except CoopIsUnavailable:
            report('unknown', '活动协同入口未确认', retryable=True)

        # 若协同未开启则调整延迟时间
        # if self.config.Event_Coop and coop_reschedule:
        #     self.config.Scheduler_ServerUpdate = '04:00, 16:00'
        self.config.task_delay(schedule=True)
