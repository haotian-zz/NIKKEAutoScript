from collections import deque
from datetime import time

from module.base.button import Button
from module.base.timer import Timer
from module.base.utils import publish_preview_frame
from module.device.win.app_control import AppControl
from module.device.win.automation import Automation
from module.exception import (
    GameNotRunningError,
    GameStuckError,
    GameTooManyClickError,
    RequestHumanTakeover,
)
from module.logger import logger


class Device(AppControl, Automation):
    # 尝试检测的 Button 集合
    detect_record = set()
    # 点击过的 Button 队列
    click_record = deque(maxlen=15)
    # 操作计时器
    stuck_timer = Timer(360, count=60).start()
    stuck_timer_long = Timer(480, count=180).start()
    """ 
        如果 detect_record 含有在 stuck_long_wait_list 中的 Button，在 stuck_timer_long 到达上限前不会 raise exception
        detect_record 值为 str(Button)，在 Button 类中，重写为该 asset 的名称
    """
    stuck_long_wait_list = ['LOGIN_CHECK', 'PAUSE']

    def __init__(self, *args, **kwargs):
        for trial in range(4):
            try:
                super().__init__(*args, **kwargs)
                break
            except GameNotRunningError:
                if trial >= 3:
                    logger.critical('Failed to start game after 3 trial')
                    raise RequestHumanTakeover
                # Try to start game
                self.current_window = self.game
                if not self.switch_to_program():
                    self.app_start()
                else:
                    # TODO 安装检查
                    break
                    # logger.critical(
                    #     f'No process "{self.config.PCClientInfo_GameProcessName}" found, please start game first'
                    # )
                    # raise RequestHumanTakeover

    def screenshot(self):
        """取得可交给任务的画面，先关闭剧情，最多等待一分钟并支持用户停止。

        剧情按钮防抖时仍继续截图，避免旧的四帧上限把未关闭的剧情泄漏给导航。
        只在检测到剧情后建立等待保护，正常截图不额外创建定时器或调用 OCR。
        """
        from module.handler.story_priority import skip_story_frame
        from module.base.watchdog import WaitGuard
        story_guard = None
        while True:
            super().screenshot()
            self.image = self.current_window.image
            if not skip_story_frame(self):
                break
            if story_guard is None:
                story_guard = WaitGuard(self, '跳过剧情', timeout=60, warn_after=15)
            story_guard.check()
            stop = getattr(self.config, 'stop_event', None)
            if stop is not None and stop.is_set():
                from module.config.config import TaskEnd
                raise TaskEnd
            self.sleep(0.2)
        guard = getattr(self, '_execution_guard', None)
        if guard is not None:
            guard.check()
        self.stuck_record_check()
        self.image = self.current_window.image
        publish_preview_frame(self.image)
        return self.image

    def handle_control_check(self, button: Button):
        """
        当点击(匹配到)Button时，清空尝试匹配过的按钮，重置操作计时器，并记录此Button，再检查点击过的Buttons

        Args:
            button: Button
        """
        self.stuck_record_clear()
        self.click_record_add(button)
        self.click_record_check()

    def click_record_check(self):
        """
        检查点击过的Buttons

        Raises:
            GameTooManyClickError:
        """
        count = {}
        for key in self.click_record:
            """
                当click_record为 ['button','button'] 时
                
                round 1:
                    count['button'] = count.get('button', default=0) + 1
                                        ↓
                    count['button'] = count.get('button', default=0) => 0 + 1
                    
                round 2:                ↓
                    count['button'] = count.get('button', default=0) => 1 + 1
                    
                count[key] = count.get(key, default=0) + 1 添加参数名 'default' 会 raise TypeError: dict.get() takes no keyword arguments
            """
            count[key] = count.get(key, 0) + 1
        count = sorted(count.items(), key=lambda item: item[1])
        if count[0][1] >= 12:
            logger.warning(f'Too many click for a button: {count[0][0]}')
            logger.warning(f'History click: {[str(prev) for prev in self.click_record]}')
            self.click_record_clear()
            raise GameTooManyClickError(f'Too many click for a button: {count[0][0]}')
        if len(count) >= 2 and count[0][1] >= 6 and count[1][1] >= 6:
            logger.warning(f'Too many click between 2 buttons: {count[0][0]}, {count[1][0]}')
            logger.warning(f'History click: {[str(prev) for prev in self.click_record]}')
            self.click_record_clear()
            raise GameTooManyClickError(f'Too many click between 2 buttons: {count[0][0]}, {count[1][0]}')

    def click_record_add(self, button: Button):
        """
        记录点击过的button

        Args:
            button: Button
            str(button): 值默认为asset名称
        """
        self.click_record.append(str(button))

    def click_record_clear(self):
        """
        清空点击过的button
        """
        self.click_record.clear()

    def stuck_record_check(self):
        """
        当操作计时器: stuck_timer，stuck_timer_long 到达限制时间时 raise exception

        如果 detect_record 含有在 stuck_long_wait_list 中的 Button，在 stuck_timer_long 到达上限前不会 raise exception
        detect_record 值为 str(Button)，在 Button 类中，默认重写为该 asset 的名称

        Raises:
            GameStuckError:
        """
        reached = self.stuck_timer.reached()
        reached_long = self.stuck_timer_long.reached()

        if not reached:
            return False
        if not reached_long:
            for button in self.stuck_long_wait_list:
                if button in self.detect_record:
                    return False

        logger.warning('Wait too long')
        logger.warning(f'Waiting for {self.detect_record}')
        self.stuck_record_clear()

        # from module.ui.ui import UI

        # ui = UI(self.config, device=self)
        # if ui.ui_additional():
        #     return False

        if self.app_is_running():
            raise GameStuckError('Wait too long')
        else:
            raise GameNotRunningError('Game died')

    def stuck_record_clear(self):
        """
        清空尝试匹配过的按钮，重置操作计时器
        """
        self.detect_record = set()
        self.stuck_timer.reset()
        self.stuck_timer_long.reset()

    def disable_stuck_detection(self):
        """
        Alas: Disable stuck detection and its handler. Usually uses in semi auto and debugging.
        禁用检查点击，操作计时器，这样在卡住时不会有任何响应
        """
        logger.info('Disable stuck detection')

        def empty_function(*arg, **kwargs):
            return False

        self.click_record_check = empty_function
        self.stuck_record_check = empty_function

    def stuck_record_add(self, button: Button):
        """
        记录尝试匹配(未匹配)的button，click_record_add 为点击过(匹配到)的button

        Args:
            button: Button
        """
        self.detect_record.add(str(button))

    def app_start(self):
        """
        启动NIKKE
        """
        super().app_start()
        self.stuck_record_clear()
        self.click_record_clear()

    def app_stop(self, program='Game'):
        """
        停止NIKKE
        """
        super().app_stop(program=program)
        self.stuck_record_clear()
        self.click_record_clear()
