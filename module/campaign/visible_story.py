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
from module.campaign.clearance import ClearanceJournal
from module.campaign.sweep import ChapterSweep, SweepUnconfirmed
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.story_marker import story_markers
from module.campaign.navigation.camera_navigation import pan_scene
from module.logger import logger

QUEST_AREA = (1565, 88, 1710, 114)
ENTRY_AREA = (570, 465, 1195, 625)
ENTRY_WAIT_SECONDS = 20.
ENTRY_READ_SECONDS = 10.
QUEST_JUMP_AREA = (1726, 80, 1769, 119)


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
    if len(stage) == 4 and stage[2] == 'EX':
        return f'{stage[0]}-EX#{stage[3]}'
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


def read_ex_entry(field, reader):
    """EX 备战同时核对独立标题模板与完整 OCR 标题，不凭共有 STAGE 字样认领 EX。

    EX 标题没有普通数字关卡身份，调用方只记录当前章及内部遇敌序号，不虚构游戏编号。
    """
    if not battle.ex_stage(field):
        return False
    x1, y1, x2, y2 = ENTRY_AREA
    crop = cv2.resize(field[y1:y2, x1:x2], None, fx=2, fy=2)
    text = reader.ocr(crop, direct_ocr=True, threshold=.9)['text']
    return bool(re.search(r'(?<![A-Z])EX\s*STAGE(?![A-Z])', text, re.I))


def quest_jump_point(field):
    """在主线任务行右侧核对蓝底白色右箭头；仅返回实测控件中心，不提供固定落点兜底。

    客户区固定为1776×999；白色连通块须为8～26px右箭头，归一化轮廓与通用几何交并比至少0.75。
    调用方还须连续核对当前主线文字；本控件只用于请求游戏定位任务，不能证明目标身份或清怪完成。
    """
    x1, y1, x2, y2 = QUEST_JUMP_AREA
    hsv = cv2.cvtColor(field[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    if np.mean(cv2.inRange(hsv, (85, 100, 100), (118, 255, 255)) > 0) < .25:
        return None
    white = cv2.inRange(hsv, (0, 0, 210), (179, 80, 255))
    template = np.zeros((32, 32), np.uint8)
    cv2.fillPoly(template, [np.array([[0, 8], [15, 8], [15, 0], [31, 15],
                                     [15, 31], [15, 23], [0, 23]])], 255)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not (8 <= width <= 26 and 8 <= height <= 26):
            continue
        glyph = cv2.resize(white[y:y + height, x:x + width], (32, 32), interpolation=cv2.INTER_NEAREST)
        if np.logical_and(glyph, template).sum() / max(1, np.logical_or(glyph, template).sum()) >= .75:
            points.append((x1 + x + width / 2, y1 + y + height / 2))
    return points[0] if len(points) == 1 else None


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
        is_ex = len(self.stage) == 4 and self.stage[2] == 'EX'
        if is_ex:
            current = self.stage if read_ex_entry(field, self.reader) else None
        else:
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
    """扫描全部敌人并核对实际遇敌身份；胜利后留在地图继续，换章前必须完成清怪巡检。"""

    def __init__(self, owner, session, executor):
        """观察簿绑定当前执行的证据目录，隔离实例及历史截图；不跨账号复用胜利或旧坐标。"""
        self.owner, self.session, self.executor = owner, session, executor
        self.session.accept_ex_entry = True
        self.memory = SceneEnemyMemory(session.evidence.folder / 'scene-enemies.json', owner.chapter_ocr)
        self.deferred = set()
        self.quest = None
        self.ex_sequence = 0
        self.manual_ex_seen = False
        self.guided_quests = set()
        self.marker_pans = 0
        self.journal = ClearanceJournal(owner.config.config_name)
        self.journal.begin(owner.current_chapter)

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

        当前章节普通、字母分支及 EX 均可识别；EX 记录内部遇敌序号，不虚构数字关卡号。
        """
        deadline = time.monotonic() + ENTRY_READ_SECONDS
        previous = None
        self.ex_sequence += 1
        ex_identity = (self.owner.current_chapter, 0, 'EX', self.ex_sequence)
        while time.monotonic() < deadline:
            self.session.check()
            field = self.session.win.client()
            if goto.battle_popup_score(field) <= .8:
                raise StoryVerificationError('核对关卡时备战弹窗消失，未开始战斗。')
            if battle.ex_stage(field):
                current = ex_identity if read_ex_entry(field, self.owner.chapter_ocr) else None
            else:
                current = read_stage(field, self.owner.chapter_ocr)
            self.session.record_evidence('scene_entry', field, actual=current, quest=self.quest)
            if current is not None and current == previous:
                return current
            previous = current
            self.session.pause(.25)
        self.close_entry()
        raise StoryVerificationError('备战关卡十秒内未能连续核对，已关闭并停止，未开始战斗。')

    def fight_entry(self, target=None):
        """按实际遇敌身份接管普通及分支；EX 保留待清，已确认胜利或其他章节只关闭。

        字母分支属于当前章节清怪范围。一次点击可能在途中触发其他敌人，因此观察簿记录的是
        encountered_on_route 关系；不会把被点击的远端敌人直接标成已清除。
        """
        stage = self.inspect_entry()
        identifier = None if target is None else target['id']
        if stage[0] != self.owner.current_chapter:
            self.close_entry()
            raise StoryVerificationError(f'备战属于其他章节 {stage_text(stage)}，已关闭并停止。')
        if len(stage) == 4 and stage[2] == 'EX':
            self.manual_ex_seen = True
            self.memory.identify(identifier, stage, 'manual_required')
            self.close_entry()
            logger.info('EX 需要手动瞄准及爆裂控制，已关闭备战并保留未完成，继续普通／分支清怪。')
            return False
        if stage in self.owner.story_wins:
            self.memory.identify(identifier, stage, 'deferred')
            self.close_entry()
            logger.info(f'已关闭备战 {stage_text(stage)}：本次已确认胜利，继续扫描其他敌人。')
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
        self.marker_pans = 0
        return True

    def scan(self, field):
        """记录两张新帧的全部红圈，仅返回稳定目标；任务读不到时仍保留独立备战身份校验。"""
        first = self.memory.observe(field)
        self.session.pause(.25)
        fresh = self.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            return fresh, []
        second = self.memory.observe(fresh)
        arrays = dict(field=fresh) if not any(item.get('kind') != 'ex' for item in second) else None
        self.session.record_evidence('scene_scan', fresh, quest=self.quest, enemies=second, arrays=arrays)
        return fresh, stable_candidates(first, second)

    def selectable(self, item):
        """过滤本轮暂不可达和需手动战斗的 EX；被过滤目标仍保留在巡检与观察簿中。

        EX 只凭高分完整字样和脚圈分类；未知类型仍通过实际备战核对，不能当作普通关卡直接开战。
        """
        if item.get('kind') == 'ex':
            if not self.manual_ex_seen:
                logger.info('发现 EX 待清目标：当前不支持手动战斗，保留未完成并优先清理普通／分支。')
            self.manual_ex_seen = True
            return False
        return (item['id'] not in self.deferred
                and item['status'] not in ('deferred', 'ex_skipped', 'unreachable', 'manual_required'))

    def guide_quest(self, field):
        """无可选普通怪时，最多一次请求游戏定位当前任务，再从新帧扫描而不猜测敌人坐标。

        同一完整任务号只尝试一次；两帧任务身份和右箭头都须一致。定位之后旧相机关联作废，
        若途中遭遇其他怪，主循环仍核对实际备战编号；此操作不替代本章末尾的清怪巡检。
        """
        if (self.quest is None or self.quest[0] != self.owner.current_chapter
                or self.quest in self.guided_quests or self.quest in self.owner.story_wins):
            return False
        self.guided_quests.add(self.quest)
        point = quest_jump_point(field)
        if point is None:
            return False
        self.session.pause(.25)
        fresh = self.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            return True
        current = read_stage(fresh, self.owner.chapter_ocr, quest=True)
        confirmed = quest_jump_point(fresh)
        if current != self.quest or confirmed is None or np.linalg.norm(np.asarray(point) - confirmed) > 5.:
            return False
        self.session.begin_target('quest_guide')
        self.session.record_evidence('quest_guide', fresh, quest=self.quest, control=confirmed)
        self.memory.previous = None
        try:
            if not self.session.tap(confirmed, interval=2.):
                return False
        except TargetTriggered:
            return True
        logger.info(f'已请求游戏定位主线任务 {stage_text(self.quest)}，下一步重新扫描全部普通／分支红圈。')
        self.session.pause(1.5)
        return True

    def reveal_marked_area(self, field):
        """用稳定的场内／离屏菱形露出目标区域，不把屏幕标记当作地面落点或敌人身份。

        同一战斗间隔最多四次，拖动不超过480px。场内／离屏分类会随指针动画变化，
        最多六帧内两次同位置识别才用于找镜头方向，不直接点击。小队精定位或离屏标记须明确，
        实际相机手势继续经过现有按下点避让；操作后必须重新扫描脚圈并核对真实备战。
        """
        if self.marker_pans >= 4:
            return False
        previous, observations, marker = [], [], None
        fresh = field
        for index in range(6):
            if index:
                self.session.pause(.25)
                fresh = self.map_frame()
                if goto.battle_popup_score(fresh) > .8:
                    return True
            observed = story_markers(fresh)
            observations.append(observed)
            markers = [item for item in observed if item['kind'] in ('scene', 'edge', 'cue')]
            confirmed = [item for item in markers if sum(
                np.linalg.norm(np.asarray(item['position']) - old['position']) <= 10. for old in previous) == 1]
            if confirmed:
                marker = max(confirmed, key=lambda item: item['confidence'])
                break
            if markers:
                previous = markers
        self.session.record_evidence('marker_probe', fresh, observations=observations, attempts=self.marker_pans,
                                     arrays=dict(field=fresh))
        logger.info(f'任务标记观测：{observations}。')
        if marker is None:
            return False
        target = np.asarray(marker['position'], float)
        drag = np.asarray([888., 450.]) - target
        length = np.linalg.norm(drag)
        if length < 120.:
            return False
        arrow = squad_arrow(fresh)
        if arrow is None and squad_offscreen(fresh) is None:
            return False
        ground = None if arrow is None else np.asarray(arrow) + [0., goto.ARROW_GROUND_OFFSET]
        drag *= min(1., 480. / length)
        self.session.begin_target('marker_reveal')
        self.session.record_evidence('marker_reveal', fresh, marker=marker, drag=drag.tolist())
        self.marker_pans += 1
        pan_scene(self.session, drag, squad_ground=ground)
        logger.info('已向稳定任务菱形调整镜头，重新扫描真实脚圈，不将菱形作为战斗落点。')
        return True

    def restore_chapter(self, current):
        """游戏已自动换章但旧章未完成巡检时，逐章返回旧章；不把新章入口当作旧章清空证据。

        复用正式选章按钮坐标，每步重新核对 NORMAL 章节号，三十秒内未恢复则停止。
        """
        deadline = time.monotonic() + 30.
        while current[0] > self.owner.current_chapter and time.monotonic() < deadline:
            if current[1] != 'normal':
                raise SweepUnconfirmed('换章后难度变化，不能复用旧章清怪状态。')
            self.session.tap((1551, 940), interval=2.)
            self.session.pause(1.)
            field = self.map_frame()
            observed = chapter_identity(field, self.owner.chapter_ocr)
            if observed is not None:
                current = observed
        if current != (self.owner.current_chapter, 'normal'):
            raise SweepUnconfirmed('未能返回尚未完成巡检的章节，清怪状态保持未完成。')
        self.memory.previous = None
        self.deferred.clear()

    def advance_chapter(self, proof):
        """本章完整巡检无剩余敌人且任务已进入下一章后，才点击下一章节并核对新身份。"""
        if proof.get('state') != 'verified_empty' or self.quest is None or self.quest[0] <= self.owner.current_chapter:
            raise SweepUnconfirmed('本章清怪或主线剧情未同时完成，保持在当前章。')
        self.journal.complete(self.owner.current_chapter, proof)
        if not getattr(self.owner, 'continuous', False):
            return dict(state='complete', reason='chapter_cleared', clearance=proof)
        self.session.tap((1715, 940), interval=2.)
        deadline = time.monotonic() + 30.
        previous = None
        while time.monotonic() < deadline:
            field = self.map_frame()
            current = chapter_identity(field, self.owner.chapter_ocr)
            if current is not None and current == previous and current == (self.owner.current_chapter + 1, 'normal'):
                logger.info(f'本章清怪巡检通过后已确认进入第 {current[0]} 章。')
                return dict(state='continue', reason='chapter_changed', clearance=proof)
            previous = current
            self.session.pause(.35)
        raise SweepUnconfirmed('本章清怪已确认，但下一章入口未能核对；已停止。')

    def run(self, ticks=48):
        """优先清理全部可见敌人，视野内目标耗尽后逐行巡检全章，清空复核通过后才换章。

        关卡顺序只作为辅助信息：已知遇敌编号从小到大、未知目标按距离排序；每次开战仍独立
        核对实际编号。总时限、单目标输入预算、停止信号及首次战败终止继续由正式会话执行。
        """
        for _ in range(ticks):
            self.session.begin_target('scene_scan')
            field = self.map_frame()
            if goto.battle_popup_score(field) > .8:
                self.fight_entry()
                continue
            identity = chapter_identity(field, self.owner.chapter_ocr)
            if identity is not None and identity[0] > self.owner.current_chapter:
                self.restore_chapter(identity)
                continue
            current = read_stage(field, self.owner.chapter_ocr, quest=True)
            if current is not None:
                if current != self.quest:
                    self.memory.reconsider(current, self.owner.story_wins)
                    self.deferred.clear()
                self.quest = current
            fresh, candidates = self.scan(field)
            if goto.battle_popup_score(fresh) > .8:
                self.fight_entry()
                continue
            candidates = [item for item in candidates if self.selectable(item)]
            if not candidates:
                if self.guide_quest(fresh):
                    continue
                if self.reveal_marked_area(fresh):
                    continue
                survey = ChapterSweep(self).run()
                if survey['state'] == 'entry':
                    self.fight_entry()
                    continue
                if survey['state'] == 'verified_empty':
                    return self.advance_chapter(survey['proof'])
                fresh, candidates = survey['field'], survey['targets']
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
            else:
                self.memory.identify(target['id'], None, 'unreachable')
                logger.info('敌人落点二十秒内未进入备战，记录暂不可达并扫描其他目标。')
                self.deferred.add(target['id'])
        return dict(state='continue', reason='tick_budget')


def run_visible_story(owner):
    """在加载底图前运行清怪闭环；所有退出路径释放会话，未完成巡检不得声明本章清空。"""
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
