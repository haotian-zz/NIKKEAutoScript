"""推图世界状态：道路版本号加按版本失效的尝试记录，替代主循环里各自清空的一次性标记。

道路可能因三类事件改变：打赢一场（win）、机关原位确认开启（switch）、踩过机关或走过平台（facility）。
「走不到」「看过」「试过」只在道路没变时有效；事件发生后对应记录自动失效，不再由各处手动清空。
用户要求（2026-10-22）：碰到开关前可以留标记，一旦触碰开关就全部重置，之前走不到的地方都可能已经能到了。
"""

# 任何道路事件都让记录失效。
ROAD = ('win', 'switch', 'facility')
# 只有真实进展（胜利、开关确认开启）才让记录失效；走过设施不算，避免设施尝试预算被自己重置。
PROGRESS = ('win', 'switch')
# 永不失效（例如「这个任务已用菱形走过一次」）。
NEVER = ()


class WorldModel:
    """记录道路事件序列与各类尝试；listener 在事件发生时清理外部缓存（如被拒绝的点击位置）。"""

    def __init__(self):
        self.history = []
        self.marks = {}
        self.listeners = []

    @property
    def epoch(self):
        """道路版本号：已发生的道路事件数。"""
        return len(self.history)

    def advance(self, reason):
        """登记一次道路事件；reason 为 win／switch／facility。"""
        if reason not in ROAD:
            raise ValueError(f'未知道路事件：{reason}')
        self.history.append(reason)
        for listener in self.listeners:
            listener(reason)

    def valid(self, epoch, reset_on):
        """在 epoch 时写下的记录，在之后的事件里是否仍然有效。"""
        return not any(reason in reset_on for reason in self.history[epoch:])

    def get(self, kind, key=None, reset_on=ROAD, default=None):
        """读取一条仍然有效的记录；失效或不存在返回 default。"""
        mark = self.marks.get((kind, key))
        if mark is None or not self.valid(mark[0], reset_on):
            return default
        return mark[1]

    def put(self, kind, key, value):
        """写入记录，版本取当前道路版本号。"""
        self.marks[(kind, key)] = (self.epoch, value)
        return value

    def once(self, kind, key=None, reset_on=ROAD):
        """同一道路状态下第一次调用返回 True 并记下，之后返回 False。"""
        if self.get(kind, key, reset_on, False):
            return False
        self.put(kind, key, True)
        return True

    def count(self, kind, key=None, reset_on=ROAD):
        """仍有效的计数；失效按 0。"""
        return self.get(kind, key, reset_on, 0)

    def bump(self, kind, key=None, reset_on=ROAD):
        """计数加一并返回新值；记录失效后从 1 重新开始。"""
        mark = self.marks.get((kind, key))
        value = mark[1] + 1 if mark is not None and self.valid(mark[0], reset_on) else 1
        self.marks[(kind, key)] = (mark[0] if mark is not None and self.valid(mark[0], reset_on)
                                   else self.epoch, value)
        return value

    def members(self, kind, key=None, reset_on=ROAD):
        """仍有效的集合记录（只读副本）。"""
        return set(self.get(kind, key, reset_on, ()))

    def add_member(self, kind, key, member, reset_on=ROAD):
        """往集合记录里加一项；记录失效后从空集合开始。"""
        members = self.members(kind, key, reset_on) | {member}
        mark = self.marks.get((kind, key))
        epoch = mark[0] if mark is not None and self.valid(mark[0], reset_on) else self.epoch
        self.marks[(kind, key)] = (epoch, members)
        return members
