import re

from module.execution.context import plan_steps, report, step
from module.base.watchdog import WaitGuard
from module.base.langs import Langs
from module.base.timer import Timer
from module.base.utils import point2str
from module.logger import logger
from module.ocr.ocr import Ocr
from module.outpost.assets import *
from module.ui.assets import COMMISSION_CHECK
from module.ui.page import page_commission
from module.ui.ui import UI


class NoOpportunity(Exception):
    pass


class Commission(UI):
    @property
    def shop_delay_list(self) -> list[str]:
        """
        妮姬列表，格式: nikke1 > nikke2 > nikke3
        """
        priority = self.config.CollectionItems_Priority
        priority = re.sub(r'\s+', '', priority).split('>')
        return [i for i in priority if i]

    def get_item_button(self, item: str):
        """
        根据选项名称获取对应的按钮
        """
        button_name = f'FAVORITE_ITEM_{item.upper()}'
        try:
            return globals()[button_name]
        except KeyError:
            logger.error(f"Button asset '{button_name}' not found for option '{item}'")
            raise

    @property
    def favorite_item_num(self):
        model_type = self.config.Optimization_OcrModelType

        ITEM_NUM = Ocr(
            [(0, 250, 720, 700)],
            name='FAVORITE_ITEM',
            model_type=model_type,
            lang='ch',
        )

        text = ITEM_NUM.ocr(self.device.image)['text']
        number = 0
        match = re.search(rf'{Langs.FAVORITE_ITEM_NUM}[:：]\s*(\d+)', text)
        if match:
            number = int(match.group(1))

        return number

    def receive(self):
        guard = WaitGuard(self.device, '派遣公告栏领取', timeout=60)
        ocr_timer = Timer(3).start()
        claimed = False
        claim_clicked = False
        logger.hr('Receive commission', 2)
        confirm_timer = Timer(3, count=5)
        click_timer = Timer(0.3)

        while 1:
            self.device.screenshot()
            guard.check()

            if self.handle_reward(interval=1):
                logger.info('Commission: reward popup received')
                claimed = True
                click_timer.reset()
                continue

            # 全部领取
            if not claim_clicked and click_timer.reached() and self.appear_then_click(
                CLAIM, threshold=20, interval=1
            ):
                claim_clicked = True
                logger.info('Commission: clicked collect all')
                click_timer.reset()
                continue

            if not claim_clicked and ocr_timer.reached() and self.appear(COMMISSION_CHECK, offset=10):
                ocr_timer.reset()
                location = self.appear_text('全部领取', threshold=0.9)
                if location and location[1] > 850:
                    x, y = map(int, location)
                    pixel = self.device.image[max(0, y - 15):y + 15, max(0, x - 50):x + 50]
                    # Enabled claim buttons have a blue fill; disabled labels must not be spammed.
                    if pixel.size and (pixel[:, :, 2].astype(float) - pixel[:, :, 0]).mean() > 30:
                        claim_clicked = True
                        logger.info('Commission: clicked collect all via recognized label')
                        self.device.click_xy(x, y)
                        continue

            # 领取完成弹出的提示
            if self.appear(CLAIM_DONE_NOTICE, offset=30):
                logger.info('Receive commission done')
                break

            # 等待领取完毕
            if self.appear(COMMISSION_CHECK, offset=10) and self.appear(CLAIM_DONE, threshold=20):
                if not confirm_timer.started():
                    confirm_timer.start()
                if confirm_timer.reached():
                    # 需要派遣
                    if self.appear(DISPATCH, threshold=10):
                        logger.info('Receive commission done')
                        break
                    # 没次数/进行中
                    if self.appear(DISPATCH_DONE, threshold=10):
                        report('success', '派遣奖励已全部领取，派遣正在进行或今日已完成' if claimed else
                               '派遣奖励已检查，无待领取项目，派遣正在进行或今日已完成')
                        return False
            else:
                confirm_timer.clear()

        report('success', '派遣公告栏奖励已领取' if claimed else '派遣公告栏已检查，暂无可领取奖励')
        return True

    def select_favorite(self, skip_first_screenshot=True):
        logger.hr('Select favorite item', 2)
        click_timer = Timer(0.3)

        # 获取当前收藏品数量
        num = 0
        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            if self.appear(COMMISSION_CHECK, offset=10) and self.appear_then_click(
                ITEM_SELECTED, threshold=20, interval=1
            ):
                click_timer.reset()
                continue

            if self.appear(ITEM_SELECTED_NUM_CHECK, offset=200):
                self.device.sleep(0.5)
                self.device.screenshot()
                num = self.favorite_item_num

                # 关闭收藏品窗口
                while 1:
                    self.device.screenshot()

                    if not self.appear(ITEM_SELECTED_NUM_CHECK, offset=200) and self.appear(
                        COMMISSION_CHECK, offset=10
                    ):
                        click_timer.reset()
                        break
                    if self.appear(ITEM_SELECTED_NUM_CHECK, offset=200):
                        logger.info('Click %s @ %s' % (point2str(10, 10), 'CLOSE_ITEM'))
                        self.device.click_xy(10, 10)
                        self.device.sleep(0.5)
                        click_timer.reset()
                        continue

                click_timer.reset()
                break

        nikke_list = self.shop_delay_list.copy()
        if not nikke_list:
            logger.info('Favorite item priority list empty, skip reselect')
            return

        need_reselect = num >= 160
        remove_current = num >= 160
        list_opened = False
        if not need_reselect:
            # 获取当前收藏品并与首位优先级比较
            while 1:
                self.device.screenshot()

                if self.appear(COMMISSION_CHECK, offset=10) and self.appear_then_click(ITEM_SELECT, offset=(100, 10)):
                    click_timer.reset()
                    continue
                if self.appear(ITEM_LIST_CHECK, offset=(70, 10)) and self.appear(
                    ITEM_LIST_SELECT_CONFIRM, threshold=10
                ):
                    list_opened = True
                    click_timer.reset()
                    break

            current = None
            for item in nikke_list:
                if self.appear(self.get_item_button(item), offset=10, static=False) and self.appear(
                    ITEM_SELECTED_IN_FAVORITE_LIST, offset=10, static=False
                ):
                    item_loc = self.appear_location(self.get_item_button(item), offset=10, static=False)
                    selected_loc = self.appear_location(ITEM_SELECTED_IN_FAVORITE_LIST, offset=10, static=False)
                    if abs(item_loc[1] - selected_loc[1]) < 50:
                        current = item
                        logger.info(f'Current favorite item: {current}')
                        break

            if current == nikke_list[0]:
                logger.info(
                    f'Current favorite item num: {num}, current item matches priority first ({current}), skip reselect'
                )
                # 退出选择列表
                while 1:
                    self.device.screenshot()
                    if self.appear(ITEM_LIST_CHECK, offset=(70, 10)) and self.appear_then_click(
                        ITEM_LIST_SELECT_CONFIRM, threshold=10
                    ):
                        click_timer.reset()
                        continue
                    if self.appear(COMMISSION_CHECK, offset=10):
                        break
                return
            if current:
                logger.info(
                    f'Current favorite item num: {num}, current item {current} != priority first {nikke_list[0]}, need reselect'
                )
                need_reselect = True
            else:
                logger.warning('Current favorite item can not judge, force reselect')
                need_reselect = True

        if need_reselect:
            # 更换收藏品
            logger.info(f'Current favorite item num: {num}, need reselect')
            # 打开收藏品列表
            if not list_opened:
                while 1:
                    self.device.screenshot()

                    if self.appear(COMMISSION_CHECK, offset=10) and self.appear_then_click(
                        ITEM_SELECT, offset=(100, 10)
                    ):
                        click_timer.reset()
                        continue
                    if self.appear(ITEM_LIST_CHECK, offset=(70, 10)) and self.appear(
                        ITEM_LIST_SELECT_CONFIRM, threshold=10
                    ):
                        click_timer.reset()
                        break

            # 循环判断超出数量的是哪个收藏品
            current = None
            for item in self.shop_delay_list:
                if self.appear(self.get_item_button(item), offset=10, static=False) and self.appear(
                    ITEM_SELECTED_IN_FAVORITE_LIST, offset=10, static=False
                ):
                    item_loc = self.appear_location(self.get_item_button(item), offset=10, static=False)
                    selected_loc = self.appear_location(ITEM_SELECTED_IN_FAVORITE_LIST, offset=10, static=False)
                    if abs(item_loc[1] - selected_loc[1]) < 50:
                        current = item
                        logger.info(f'Current favorite item: {current}')
                        break
            if not current:
                logger.warning('Current favorite item can not judge')
            else:
                # 只有多个候选时才移除
                if remove_current and len(nikke_list) > 1 and current in nikke_list:
                    nikke_list.remove(current)
                    logger.info(f'Remove current favorite item: {current}')
                    self.config.CollectionItems_Priority = ' > '.join(nikke_list)
                else:
                    logger.info(f'Keep current favorite item: {current}')

            # 滑动到列表开始位置
            self.ensure_sroll((620, 400), (620, 900), speed=30, count=3, delay=0.3, method='swipe')
            # 选择新的收藏品
            while 1:
                self.device.screenshot()

                # 派遣主页
                if self.appear(COMMISSION_CHECK, offset=10):
                    break

                # 选择收藏品（这里取的是删除旧收藏品之后的第一个）
                if self.appear(ITEM_LIST_CHECK, offset=(70, 10)) and nikke_list:
                    if self.appear_then_click(
                        self.get_item_button(nikke_list[0]), offset=10, click_offset=(150, 0), interval=1, static=False
                    ):
                        logger.info(f'Select new favorite item: {nikke_list[0]}')
                        click_timer.reset()
                    else:
                        # 未找到收藏品，进行滑动
                        self.device.sleep(0.5)
                        self.ensure_sroll((620, 1000), (620, 700), speed=5, count=1, delay=0.5, method='scroll')
                        continue

                # 派遣选择
                if self.appear(ITEM_LIST_CHECK, offset=(70, 10)) and self.appear_then_click(
                    ITEM_LIST_SELECT_CONFIRM, threshold=10
                ):
                    click_timer.reset()
                    continue

    def dispatch(self, skip_first_screenshot=True):
        guard = WaitGuard(self.device, '派遣公告栏派出', timeout=60)
        logger.hr('Dispatch commission', 2)
        dispatch_clicked = False
        click_timer = Timer(0.3)

        while 1:
            if skip_first_screenshot:
                skip_first_screenshot = False
            else:
                self.device.screenshot()

            guard.check()

            # 派遣
            if (
                not dispatch_clicked
                and click_timer.reached()
                and self.appear(COMMISSION_CHECK, offset=10)
                and self.appear_then_click(DISPATCH, offset=10, interval=1)
            ):
                dispatch_clicked = True
                logger.info('Commission: clicked dispatch all')
                click_timer.reset()
                continue

            # 派遣确认
            if (
                click_timer.reached()
                and self.appear(ITEM_LIST_CHECK, offset=(70, 10))  # 复用
                and self.appear_then_click(DISPATCH_CONFIRM, offset=10, interval=1)
            ):
                click_timer.reset()
                continue

            if self.appear(COMMISSION_CHECK, offset=(10, 10)) and self.appear(DISPATCH_DONE, threshold=10):
                logger.info('Dispatch commission done')
                break

    def run(self):
        logger.hr('Dispatch and claim commission')
        items = [('commission.claim', '公告栏全部领取'), ('commission.dispatch', '公告栏派出')]
        if self.config.CollectionItems_Enable and self.config.CollectionItems_Priority:
            items.insert(1, ('commission.collection', '派遣收藏品'))
        plan_steps(items)
        self.ui_ensure(page_commission)
        step('commission.claim', '公告栏全部领取', self.receive, verified=True)
        if self.config.CollectionItems_Enable and self.config.CollectionItems_Priority:
            step('commission.collection', '派遣收藏品', self.select_favorite, verified=True)
        step('commission.dispatch', '公告栏派出', self.dispatch, verified=True)
        self.config.task_delay(schedule=True)
