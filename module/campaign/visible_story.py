"""逐屏扫描敌人并按实际备战身份推进；任务提示辅助找方向，不替代关卡与胜负证据。"""

import re
import time

import cv2
import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.vision import battle
from module.campaign.vision.chapter import chapter_identity
from module.campaign.scene_memory import SceneEnemyMemory, stable_candidates
from module.campaign.vision.perception import squad_arrow
from module.logger import logger

QUEST_AREA = (1565, 88, 1710, 114)
ENTRY_AREA = (570, 465, 1195, 625)
ENTRY_WAIT_SECONDS = 20.
ENTRY_READ_SECONDS = 10.


class StoryVerificationError(RuntimeError):
    """关卡身份或战后主线进展无法确认；保留具体原因，禁止转成定位恢复重试。"""

    def __init__(self, message, reason='story_identity_unconfirmed'):
        """保存简报原因码；message 应描述当前可见问题，不包含推测的地图坐标。"""
        super().__init__(message)
        self.reason = reason


def parse_stage(text, quest=False):
    """解析完整普通或分支关卡号；20-6B-1 保留四段身份，绝不截成 20-6。

    任务栏须含通关动词，EX、残缺后缀、多个不同编号和零值均拒绝。备战页允许编号后紧跟 STAGE。
    """
    if (quest and not re.search(r'通[关關]|CLEAR', text, re.I)
            or re.search(r'(?<![A-Z])EX(?:\s*STAGE|\s*[-－–]?\s*\d)', text, re.I)):
        return None
    pattern = r'(?<!\d)([1-9]\d?)\s*[-－–]\s*([1-9]\d?)(?:\s*([A-Z])\s*[-－–]\s*([1-9]\d?))?'
    values = set()
    for match in re.finditer(pattern, text, re.I):
        tail = text[match.end():].lstrip()
        if re.match(r'\d|[-－–]|[A-Z](?!TAGE)', tail, re.I):
            return None
        chapter, stage, branch, number = match.groups()
        value = (int(chapter), int(stage))
        if branch is not None:
            value += (branch.upper(), int(number))
        values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def stage_text(stage):
    """按游戏显示格式输出完整身份；普通关卡和字母分支不共享一个简写。"""
    base = f'{stage[0]}-{stage[1]}'
    return base if len(stage) == 2 else f'{base}{stage[2]}-{stage[3]}'


def stage_order(stage):
    """将主线及其字母分支转为可比较的顺序提示；只辅助排队，不证明道路或解锁关系。"""
    return (stage[0], stage[1], '' if len(stage) == 2 else stage[2], 0 if len(stage) == 2 else stage[3])


def read_stage(field, reader, quest=False):
    """任务栏裁出亮字后直接识别单行，备战标题使用普通 OCR；均只处理当前截图。

    固定任务行不再依赖通用文字检测器，避免细小数字被拆行或漏检；保留 0.9 分和完整任务格式。
    """
    if field.shape != (999, 1776, 3):
        raise ValueError('主线关卡核对需要 1776×999 BGR 客户区。')
    x1, y1, x2, y2 = QUEST_AREA if quest else ENTRY_AREA
    crop = field[y1:y2, x1:x2]
    if not quest:
        crop = cv2.resize(crop, None, fx=2, fy=2)
        return parse_stage(reader.ocr(crop, direct_ocr=True, threshold=.9)['text'])
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ys, xs = np.nonzero((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 150))
    if len(xs) < 20:
        return None
    crop = crop[max(0, ys.min() - 2):ys.max() + 3, max(0, xs.min() - 2):xs.max() + 3]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    results = reader.paddleocr.recognize([crop])
    if len(results) != 1:
        return None
    text, score = results[0].get('rec_text', ''), float(results[0].get('rec_score', 0.))
    logger.info(f'主线任务单行识别：{text!r}，分数 {score:.3f}')
    return parse_stage(text, quest=True) if np.isfinite(score) and score >= .9 else None


class EntryStageGuard:
    """在实际开战前连续核对两帧备战关卡；错误或十秒仍读不清时关闭弹窗并停止。"""

    def __init__(self, session, stage, reader, confirmed=False):
        """计时不包含走路；confirmed 只由已连续核对两帧的入口传入，开战前仍重读新帧。"""
        self.session, self.stage, self.reader = session, tuple(stage), reader
        self.started = None
        self.previous, self.stable = (self.stage, 1) if confirmed else (None, 0)

    def __call__(self, field):
        """仅对连续正确两帧放行；关闭路径显式跳过关卡校验，避免递归。"""
        if self.started is None:
            self.started = time.monotonic()
        current = read_stage(field, self.reader)
        self.session.record_evidence('story_entry', field, expected=self.stage, actual=current)
        self.stable = self.stable + 1 if current is not None and current == self.previous else int(current is not None)
        self.previous = current
        if current == self.stage and self.stable >= 2:
            logger.info(f'备战关卡已连续确认：{stage_text(current)}，允许进入战斗。')
            return True
        if (current is not None and current != self.stage and self.stable >= 2
                or time.monotonic() - self.started >= ENTRY_READ_SECONDS):
            self.session.preview(field)
            self.session.battle_step(close_entry=True)
            raise StoryVerificationError(f'备战关卡无法核对：预期 {stage_text(self.stage)}，'
                                         f'识别到 {current}；已停止，未开始战斗。')
        return False


class VisibleStoryRunner:
    """扫描可见敌人、记录位置与实际遇敌身份；胜利后留在地图继续，困难道路才交导航。"""

    def __init__(self, owner, session, executor):
        """观察簿绑定当前执行的证据目录，隔离实例及历史截图；不跨账号复用胜利或旧坐标。"""
        self.owner, self.session, self.executor = owner, session, executor
        self.memory = SceneEnemyMemory(session.evidence.folder / 'scene-enemies.json')
        self.deferred = set()
        self.quest = None

    def map_frame(self, timeout=30.):
        """等待地图或备战弹窗，剧情出现即跳过；不在加载或未知画面发送地图落点。

        剧情处理时禁止普通备战按钮放行，避免截图之间切换成弹窗后未经身份校验就开战。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.session.check()
            field = self.session.win.client()
            if goto.battle_popup_score(field) > .8:
                return field
            if battle.field_visible(field):
                self.session.identity(field)
                return field
            state = battle.battle_state(field)['state']
            if state == 'story':
                self.session.battle_mode = True
                self.session.entry_validator = lambda frame: False
                try:
                    self.session.battle_step()
                finally:
                    self.session.battle_mode = False
                    self.session.entry_validator = None
            self.session.pause(.3)
        self.session.preview(self.session.win.client())
        raise StoryVerificationError('等待地图或备战超过三十秒；已保存画面并停止。', 'story_field_timeout')

    def wait_entry(self):
        """点击后最多等二十秒并跳过剧情；途中遇到任何普通战斗立即交给实际关卡核对。"""
        deadline = time.monotonic() + ENTRY_WAIT_SECONDS
        warned = False
        while time.monotonic() < deadline:
            field = self.map_frame(timeout=max(.1, deadline - time.monotonic()))
            if goto.battle_popup_score(field) > .8:
                return True
            if not warned and deadline - time.monotonic() < ENTRY_WAIT_SECONDS / 2:
                self.session.preview(field)
                logger.info('敌人落点尚未进入备战，已保存等待画面；不重复发送同一落点。')
                warned = True
            self.session.pause(.3)
        return False

    def close_entry(self):
        """仅关闭已识别的准备弹窗，最多五秒；所有分支始终禁止开始战斗。"""
        deadline = time.monotonic() + 5.
        self.session.battle_mode = True
        try:
            while time.monotonic() < deadline:
                field = self.session.win.client()
                if goto.battle_popup_score(field) <= .8:
                    return
                self.session.battle_step(close_entry=True)
                self.session.pause(.3)
        finally:
            self.session.battle_mode = False
        raise StoryVerificationError('备战弹窗未能关闭，已停止，未开始战斗。')

    def inspect_entry(self):
        """连续两帧读取实际遇敌完整编号；不假设它就是被点击的敌人。

        可接管当前章节普通及字母分支；EX 仅关闭。十秒仍读不清就停止，不凭任务栏猜编号。
        """
        deadline = time.monotonic() + ENTRY_READ_SECONDS
        previous = None
        while time.monotonic() < deadline:
            self.session.check()
            field = self.session.win.client()
            if goto.battle_popup_score(field) <= .8:
                raise StoryVerificationError('核对关卡时备战弹窗消失，未开始战斗。')
            if battle.ex_stage(field):
                self.close_entry()
                return None
            current = read_stage(field, self.owner.chapter_ocr)
            self.session.record_evidence('scene_entry', field, actual=current, quest=self.quest)
            if current is not None and current == previous:
                return current
            previous = current
            self.session.pause(.25)
        self.close_entry()
        raise StoryVerificationError('备战关卡十秒内未能连续核对，已关闭并停止，未开始战斗。')

    def fight_entry(self, target=None):
        """按实际遇敌身份接管战斗；已胜利、未来编号或其他章节只关闭，不消耗再次挑战。

        字母分支属于当前章节清怪范围。一次点击可能在途中触发其他敌人，因此观察簿记录的是
        encountered_on_route 关系；不会把被点击的远端敌人直接标成已清除。
        """
        stage = self.inspect_entry()
        identifier = None if target is None else target['id']
        if stage is None:
            self.memory.identify(identifier, None, 'ex_skipped')
            return False
        if stage[0] != self.owner.current_chapter:
            self.close_entry()
            raise StoryVerificationError(f'备战属于其他章节 {stage_text(stage)}，已关闭并停止。')
        if stage in self.owner.story_wins or (self.quest is not None and stage[1] > self.quest[1]):
            self.memory.identify(identifier, stage, 'deferred')
            self.close_entry()
            logger.info(f'已关闭备战 {stage_text(stage)}：本次已确认胜利或编号超过当前任务，继续扫描其他敌人。')
            return False
        self.memory.identify(identifier, stage, 'encountered_on_route')
        goal = dict(kind='story', id='story-' + stage_text(stage), stage=list(stage))
        self.session.entry_validator = EntryStageGuard(self.session, stage, self.owner.chapter_ocr, confirmed=True)
        try:
            self.executor.battle(goal)
        finally:
            self.session.entry_validator = None
            self.session.reuse_squad = False
        self.memory.victory(stage)
        self.memory.reconsider(self.quest, self.owner.story_wins, after_win=True)
        guard = getattr(self.owner.device, '_execution_guard', None)
        if guard is not None:
            guard.progress()
        logger.info(f'地图战斗已确认胜利：{stage_text(stage)}；留在地图继续扫描，不要求每次胜利都改变任务栏。')
        self.deferred.clear()
        return True

    def scan(self, field):
        """记录两张新帧的全部红圈，仅返回稳定目标；任务读不到时仍保留独立备战身份校验。"""
        first = self.memory.observe(field)
        self.session.pause(.25)
        fresh = self.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            return fresh, []
        second = self.memory.observe(fresh)
        self.session.record_evidence('scene_scan', fresh, quest=self.quest, enemies=second)
        return fresh, stable_candidates(first, second)

    def run(self, ticks=48):
        """优先清理可见敌人，目标耗尽后有限显露任务方向；道路受阻时才交现有导航。

        关卡顺序只作为辅助信息：已知遇敌编号从小到大、未知目标按距离排序；每次开战仍独立
        核对实际编号。总时限、单目标输入预算、停止信号及首次战败终止继续由正式会话执行。
        """
        from module.campaign.navigation.story_navigation import reveal_story_marker
        revealed = False
        for _ in range(ticks):
            self.session.begin_target('scene_scan')
            field = self.map_frame()
            if goto.battle_popup_score(field) > .8:
                self.fight_entry()
                continue
            identity = chapter_identity(field, self.owner.chapter_ocr)
            if identity is not None and identity[0] > self.owner.current_chapter:
                return dict(state='continue', reason='chapter_changed')
            current = read_stage(field, self.owner.chapter_ocr, quest=True)
            if current is not None and current[0] == self.owner.current_chapter:
                if current != self.quest:
                    self.memory.reconsider(current, self.owner.story_wins)
                    self.deferred.clear()
                self.quest = current
            fresh, candidates = self.scan(field)
            if goto.battle_popup_score(fresh) > .8:
                self.fight_entry()
                continue
            candidates = [item for item in candidates if item['id'] not in self.deferred
                          and item['status'] not in ('deferred', 'ex_skipped', 'unreachable')]
            if not candidates:
                if not revealed and reveal_story_marker(self.session):
                    revealed = True
                    continue
                return dict(state='fallback', message='当前可见敌人已处理或未能到达，转入道路与机关导航')
            anchor = squad_arrow(fresh)
            anchor = np.asarray(goto.VIEWPORT_PIXEL if anchor is None else anchor)
            target = min(candidates, key=lambda item: (
                stage_order(item['stage']) if item['stage'] is not None else (99, 99, '', 0),
                float(np.linalg.norm(np.asarray(item['click']) - anchor))))
            self.session.begin_target(target['id'])
            self.session.preview(fresh)
            self.session.record_evidence('scene_target', fresh, quest=self.quest, target=target)
            logger.info(f"地图敌人目标：{target['id']}，脚圈 {target['click']}，当前可选 {len(candidates)} 个；"
                        '到达后读取实际遇敌关卡。')
            try:
                self.session.tap(target['click'])
            except TargetTriggered:
                # 最后一道输入闸门可能比扫描更早读到途中弹窗；只核对实际备战，不补发旧落点。
                pending = self.map_frame()
                if goto.battle_popup_score(pending) > .8:
                    self.fight_entry()
                continue
            if self.wait_entry():
                won = self.fight_entry(target)
                if not won:
                    self.deferred.add(target['id'])
                revealed = False
            else:
                self.memory.identify(target['id'], None, 'unreachable')
                logger.info('敌人落点二十秒内未进入备战，记录暂不可达并扫描其他目标。')
                self.deferred.add(target['id'])
        return dict(state='continue', reason='tick_budget')


def run_visible_story(owner):
    """在加载底图前运行主线直接闭环；所有退出路径释放会话，地图导航仅在明确 fallback 时接手。"""
    from module.campaign.execution import CampaignExecutor
    from module.campaign.session import CampaignSession
    owner.story_wins = set()
    owner.prepare_client()
    owner.check_main_story_page()
    owner.detect_current_chapter(localize=False)
    session = CampaignSession(owner, navigation=False)
    try:
        session.start()
        return VisibleStoryRunner(owner, session, CampaignExecutor(owner, session)).run()
    finally:
        session.close()
