from typing import TYPE_CHECKING
from module.execution.context import report

from module.base.decorator import Config
from module.base.timer import Timer
from module.logger import logger

if TYPE_CHECKING:
    from module.event.event import Event


class EventExtend:
    @Config.when(EVENT_EXTEND=True)
    def event_extend(self: 'Event'):
        logger.hr('START EVENT EXTEND', 2)
        return self.event_extend_by_event_id()

    @Config.when(EVENT_EXTEND=False)
    def event_extend(self: 'Event'):
        report('skipped', '本活动没有额外内容')
        return

    @Config.when(EVENT_ID='event_20260423')
    def event_extend_by_event_id(self: 'Event'):
        logger.info('Run event extend')

        confirm_timer = Timer(1, count=3)
        while 1:
            self.device.screenshot()
            if self.appear_then_click(self.event_assets.EXTEND_BAG_CONFIRM, offset=30, interval=1):
                continue
            if self.appear_then_click(self.event_assets.EXTEND_BAG, offset=30, interval=1):
                continue
            if self.appear(self.event_assets.EXTEND_BAG_CHECK, offset=30):
                if not confirm_timer.started():
                    confirm_timer.start()
                if confirm_timer.reached():
                    break
            else:
                confirm_timer.clear()

        self.back_to_event()
        return
