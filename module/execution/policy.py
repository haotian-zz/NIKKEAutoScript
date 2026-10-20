"""Only tasks that recheck game state can be resumed from a report."""


SAFE_TASKS = {
    'Reward', 'Mailbox', 'DailyGift', 'WeeklyGift', 'MonthlyGift', 'StepUpGift', 'Commission',
    'Conversation', 'Shop', 'RubbishShop', 'SimulationRoom', 'TribeTower', 'Interception',
    'RookieArena', 'SpecialArena', 'ChampionArena', 'SoloRaid', 'Coop', 'Liberation',
    'MissionPass', 'WarehouseStats', 'Event', 'Event2', 'OutpostStory', 'EpisodeViewing',
    'Daily', 'Destruction', 'Notify',
}


def retry_policy(config, command):
    task = config.get(command, {})
    if not task.get('Scheduler', {}).get('Enable'):
        return False, '任务目前未启用，请先检查任务配置'
    if command not in SAFE_TASKS:
        return False, '此任务尚不支持安全续做，请先在游戏中核实执行结果'
    if command == 'Notify':
        return True, '通知补做仅检查开关，不会重复发送消息'
    if command == 'Daily' and any(task.get('Daily', {}).get(key) for key in ('EnhanceEquipment', 'SendGift')):
        return False, '已开启送礼或强化，请先核实资源消耗后单独执行'
    if command == 'Destruction' and task.get('Destruction', {}).get('UseGem'):
        return False, '已开启宝石歼灭，请先核实资源消耗后单独执行'
    if command in ('Event', 'Event2') and task.get('StoryStage', {}).get('HardTicket'):
        return False, '已开启活动买票，请先核实购买结果后单独执行'
    return True, ''
