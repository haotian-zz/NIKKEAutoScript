from module.base.decorator import del_cached_property
from module.execution.context import plan_steps, step
from module.shop.arena import ARENA_CODE_MANUAL_PRODUCTS, ArenaShop
from module.shop.base import (
    NotEnoughMoneyError,
    Product,
    ProductQueueIsEmpty,
    PurchaseTimeTooLong,
    Refresh,
    ShiftyShopReplaced,
    ShopBase,
)
from module.shop.body_label import BodyLabelShop
from module.shop.general import GeneralShop
from module.ui.page import page_shop

__all__ = [
    'ARENA_CODE_MANUAL_PRODUCTS',
    'NotEnoughMoneyError',
    'Product',
    'ProductQueueIsEmpty',
    'PurchaseTimeTooLong',
    'Refresh',
    'ShiftyShopReplaced',
    'ShopBase',
    'Shop',
]


class Shop(GeneralShop, ArenaShop, BodyLabelShop):
    def run(self):
        plan_steps([(key, label) for key, label, enabled in [
            ('shop.general', '普通商店', self.config.GeneralShop_enable),
            ('shop.body_label', '躯体标签商店', self.config.BodyLabelShop_enable),
            ('shop.arena', '竞技场商店', self.config.ArenaShop_enable),
        ] if enabled])
        self.ui_ensure(page_shop)
        if self.config.GeneralShop_enable:
            step('shop.general', '普通商店', self.run_general_shop, verified=True)
        if self.config.BodyLabelShop_enable:
            step('shop.body_label', '躯体标签商店', self.run_body_label_shop, verified=True)
        if self.config.ArenaShop_enable:
            step('shop.arena', '竞技场商店', self.run_arena_shop, verified=True)

        del_cached_property(self, 'general_shop_priority')
        del_cached_property(self, 'arena_shop_priority')
        self.config.task_delay(schedule=True)
