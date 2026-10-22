"""逐屏扫描敌人并按实际备战身份推进；任务提示辅助找方向，不替代关卡与胜负证据。"""

import re
import time

import cv2
import numpy as np

from module.campaign.navigation import controls as goto
from module.campaign.navigation.movement_feedback import TargetTriggered
from module.campaign.vision import battle
from module.campaign.vision.chapter import chapter_identity
from module.campaign.vision.entry_title import read_title, title_crop
from module.campaign.scene_memory import SceneEnemyMemory, stable_candidates
from module.campaign.clearance import ClearanceJournal
from module.campaign.chapter_flow import ChapterOverlayFlow
from module.campaign.sweep import ChapterSweep, SweepUnconfirmed
from module.campaign.vision.perception import squad_arrow
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.story_marker import story_markers
from module.campaign.vision.native_motion import ClickProgress, MinimapStability, NativeMotion, combine_motion
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.window import UiControlChanged
from module.logger import logger

# 目标脚圈离已记下的需手动关卡（小队触发备战时的脚下位置）多近算同一个敌人（客户区像素）：
# 小队在脚圈约28底图像素内触发备战，约75客户区像素，再留关键帧投影误差。
MANUAL_ENTRY_RADIUS = 120.

QUEST_AREA = (1565, 88, 1710, 114)
LOBBY_QUEST_AREA = (445, 108, 657, 147)
ENTRY_AREA = (570, 465, 1195, 625)
ENTRY_TITLE_AREA = (620, 535, 970, 580)
ENTRY_WAIT_SECONDS = 20.
# 原生寻路绕远路的总跟随时限；只有持续的移动证据才能延到这里。
ENTRY_MOVE_SECONDS = 90.
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


def parse_entry_title(text):
    """仅解析带完整STAGE或BOSS后缀的单行标题，保留完整字母分支编号。

    B0SS只容忍标签中字母O的常见OCR混淆，不修改编号中的数字。不接受EX、缺失后缀、
    残缺分支或额外文字；避免将宽区域OCR拼接的BOSS误当分支B，也不从期望任务补编号。
    """
    number = r'[1-9]\d?'
    separator = r'[-－–]'
    identity = rf'{number}\s*{separator}\s*{number}(?:\s*[A-Z]\s*{separator}\s*{number})?'
    match = re.fullmatch(rf'\s*({identity})\s*(?:STAGE|B[O0]SS)\s*', text, re.I)
    return parse_stage(match.group(1)) if match else None


def read_stage(field, reader, quest=False):
    """任务栏与备战标题优先识别固定单行，仅处理当前截图，不用任务号替代实际备战。

    备战先使用CPU专用识别器；未安装权重或未通过时依次尝试原单行、字形裁切与宽区域OCR。
    单行均须完整STAGE/BOSS锚点及0.9分，不补写数字。任务栏及EX的独立核对路径不变。
    """
    if field.shape != (999, 1776, 3):
        raise ValueError('主线关卡核对需要 1776×999 BGR 客户区。')
    x1, y1, x2, y2 = QUEST_AREA if quest else ENTRY_AREA
    crop = field[y1:y2, x1:x2]
    if not quest:
        title = read_title(field)
        if title is not None:
            text, score = title['rec_text'], float(title['rec_score'])
            stage = parse_entry_title(text) if np.isfinite(score) and score >= .9 else None
            logger.info(f'备战标题专用识别：{text!r}，分数 {score:.3f}')
            if stage is not None:
                return stage
        left, top, right, bottom = ENTRY_TITLE_AREA
        title = cv2.resize(field[top:bottom, left:right], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = reader.paddleocr.recognize([title])
        if len(results) == 1:
            text, score = results[0].get('rec_text', ''), float(results[0].get('rec_score', 0.))
            stage = parse_entry_title(text) if np.isfinite(score) and score >= .9 else None
            logger.info(f'备战标题单行识别：{text!r}，分数 {score:.3f}')
            if stage is not None:
                return stage
        normalized = title_crop(field, stretch=2)
        if normalized is not None:
            results = reader.paddleocr.recognize([normalized])
            if len(results) == 1:
                text, score = results[0].get('rec_text', ''), float(results[0].get('rec_score', 0.))
                stage = parse_entry_title(text) if np.isfinite(score) and score >= .9 else None
                if stage is not None:
                    return stage
        crop = cv2.resize(crop, None, fx=2, fy=2)
        return parse_stage(reader.ocr(crop, direct_ocr=True, threshold=.9)['text'])
    text = read_quest_text(field, reader)
    return parse_stage(text, quest=True) if text is not None else None


def read_quest_text(field, reader):
    """读取固定主线任务行，保留0.9分门槛；兼容1776×999地图和720×1280大厅。

    返回高分原文或None；任务类型由调用方独立解析，不从当前章节或已胜关卡补全。
    两种已知布局分别裁单行，避开图标和第二条活动任务；其他尺寸直接拒绝。
    """
    if field.shape == (999, 1776, 3):
        area = QUEST_AREA
    elif field.shape == (1280, 720, 3):
        area = LOBBY_QUEST_AREA
    else:
        raise ValueError('主线任务核对需要1776×999或720×1280客户区。')
    x1, y1, x2, y2 = area
    crop = field[y1:y2, x1:x2]
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
    return text if np.isfinite(score) and score >= .9 else None


def return_lobby_quest(text):
    """只接受完整的简繁体返回大厅任务，不从含大厅的说明句或残缺文字猜测操作。"""
    return bool(text is not None and re.fullmatch(r'返回(?:【(?:大厅|大廳)】|\[(?:大厅|大廳)\]|大厅|大廳)',
                                                re.sub(r'\s+', '', text)))


def destination_lobby_quest(text):
    """大厅「前往【某处】」任务：通关后游戏要求去某个界面触发剧情或解锁，点任务箭头由游戏带过去。

    实录 2026-10-21：21 章通关后为「前往【前哨基地】」，进入基地即完成。只接受整行“前往＋括号地名”，
    「前往地面【出击】」是出击指令，不在此列。
    """
    return bool(text is not None and re.fullmatch(r'前往(?:【[^】]{1,12}】|\[[^\]]{1,12}\]|前哨基地)',
                                                re.sub(r'\s+', '', text)))


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


def quest_jump_point(field, area=QUEST_JUMP_AREA):
    """在主线任务行右侧核对蓝底白色右箭头；仅返回实测控件中心，不提供固定落点兜底。

    默认区域用于1776×999地图，area可明确提供720×1280大厅任务区域；越界区域直接拒绝。
    白色连通块须为8～26px右箭头，归一化轮廓与通用几何交并比至少0.75。
    调用方还须连续核对当前主线文字；本控件只用于请求游戏定位任务，不能证明目标身份或清怪完成。
    """
    x1, y1, x2, y2 = area
    if not (0 <= x1 < x2 <= field.shape[1] and 0 <= y1 < y2 <= field.shape[0]):
        return None
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
        self.session.record_evidence('story_entry', field, expected=self.stage, actual=current, arrays=dict(field=field))
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
    """核对实际遇敌身份并持续推进主线；单章清理保留完整巡检，旧章可选内容保留待办。"""

    def __init__(self, owner, session, executor):
        """观察簿绑定当前执行的证据目录，隔离实例及历史截图；不跨账号复用胜利或旧坐标。"""
        self.owner, self.session, self.executor = owner, session, executor
        self.chapter_flow = ChapterOverlayFlow(session, owner.chapter_ocr)
        self.session.accept_ex_entry = True
        self.memory = SceneEnemyMemory(session.evidence.folder / 'scene-enemies.json', owner.chapter_ocr)
        self.deferred = set()
        self.quest = None
        self.ex_sequence = 0
        self.manual_ex_seen = False
        self.guided_quests = set()
        self.marker_pans = 0
        self.marker_jumps = 0
        self.blocked_views = 0
        self.structure_inputs = 0
        self.structure_pans = 0
        self.facilities = None
        self.facility_survey_attempted = False
        # 本章开局设施巡检是否已处理（每个运行器一次，是否真的巡检由目录记录决定）。
        self.chapter_surveyed = False
        # 受阻目标 → 已按底图坐标去看过的目录机关编号；踩亮开关后清空。
        self.known_switch_tries = {}
        # 已核对过是否为需手动关卡（EX）位置的目标编号。
        self.manual_checked = set()
        # 最近一次点击画面内目标时的那帧；需手动关卡按它记位置。
        self.tap_frame = None
        self.map_localizer = None
        # 受阻目标 → 已做螺旋机关搜索次数；已点过但未解决的搜索键（同状态下不重复点击）。
        self.switch_searches, self.switch_skip = {}, set()
        # 已用任务定位看过目标旁的受阻目标（每个只看一次）。
        self.goal_views = set()
        # 刚受阻、设施恢复尚未开始：先把画面内其余可见敌人逐个试一遍（可能就是挡路的支线怪）。
        self.try_visible = False
        self.gate_enemy_tries = {}
        # 受阻目标 → 螺旋下次续看的屏号（被守关敌人打断后不从第0屏重来）；踩亮开关后清空。
        self.spiral_resume = {}
        # 已连续看完全部螺旋屏幕找过机关的受阻目标；踩亮开关后清空。
        self.spiral_done = set()
        self.blocked_since, self.blocked_click = {}, None
        self.entry_outcome = None
        self.journal = ClearanceJournal(owner.config.config_name)
        self.journal.begin(owner.current_chapter)

    def map_frame(self, timeout=30., track=True):
        """等待地图或备战弹窗，剧情出现即跳过；不在加载或未知画面发送地图落点。

        剧情处理时禁止普通备战按钮放行，避免截图之间切换成弹窗后未经身份校验就开战。
        track为False时不做设施背景桥接：螺旋搜索每屏自行登记一次目录，不必每帧再配准。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.session.check()
            field = self.session.win.client()
            if self.chapter_flow.handle(field):
                continue
            if goto.battle_popup_score(field) > .8:
                return field
            if battle.field_visible(field):
                self.session.identity(field)
                if track and self.facilities is not None:
                    self.facilities.track_background(field)
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

    def wait_entry(self, before, arrow=None, target=None):
        """观察游戏原生寻路；只有真实移动才续等，停滞与观测失败分开处理。

        三次且跨2秒的可靠静止才返回False请求设施诊断。观测不足或持续移动超过总时限
        均保留未确认并停止，不能将二十秒没弹窗写成道路不可达，也不在途中追加目标点击。
        """
        progress = ClickProgress(time.monotonic(), unknown_limit=ENTRY_WAIT_SECONDS, total_limit=ENTRY_MOVE_SECONDS)
        motion = NativeMotion(before, arrow, None if target is None else target['click'])
        minimap = MinimapStability(before)
        previous_state = None
        while True:
            field = self.map_frame(timeout=max(.1, min(30., progress.started+ENTRY_MOVE_SECONDS-time.monotonic())))
            if goto.battle_popup_score(field) > .8:
                self.entry_outcome = dict(state='entry')
                return True
            observation = combine_motion(motion.observe(field, squad_arrow(field)), minimap.observe(field))
            now = time.monotonic()
            state = progress.update(now, observation)
            self.entry_outcome = dict(observation, outcome=state, elapsed=round(now-progress.started, 2))
            if observation['state'] != previous_state or state != 'waiting':
                self.session.record_evidence('native_click_feedback', field, feedback=self.entry_outcome,
                                             target=target, arrays=dict(field=field))
                observed_text = {'moving': '小队移动中', 'stationary': '小队暂时静止',
                                 'unknown': '小队或背景观测未确认'}[observation['state']]
                result_text = {'waiting': '继续观察', 'stalled': '连续停滞，进入诊断',
                               'movement_timeout': '达到本次移动总时限',
                               'observation_unknown': '未获得可靠结果，停止追加输入'}[state]
                logger.info(f'原生寻路反馈：{observed_text}，耗时{now-progress.started:.1f}秒；{result_text}。')
                previous_state = observation['state']
            if state == 'stalled':
                return False
            if state != 'waiting':
                raise StoryVerificationError('原生寻路未确认到达；已保留观测，未发送后续目标。', 'native_'+state)
            self.session.pause(.7)

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
            self.session.record_evidence('scene_entry', field, actual=current, quest=self.quest, arrays=dict(field=field))
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
            self.remember_manual_entry(stage, target)
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
        self.marker_pans = self.marker_jumps = 0
        self.blocked_views = self.structure_inputs = self.structure_pans = 0
        return True

    def scan(self, field):
        """记录两张新帧的全部红圈，仅返回稳定目标；任务读不到时仍保留独立备战身份校验。"""
        first = self.memory.observe(field)
        self.session.pause(.25)
        fresh = self.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            return fresh, []
        second = self.memory.observe(fresh)
        candidates = stable_candidates(first, second)
        if (first or second) and not candidates:
            arrays = dict(field=fresh, previous_field=field)
        else:
            arrays = dict(field=fresh) if not any(item.get('kind') != 'ex' for item in second) else None
        self.session.record_evidence('scene_scan', fresh, quest=self.quest, enemies=second, arrays=arrays,
                                     previous_enemies=first, stable=len(candidates),
                                     squad_overlap=getattr(self.memory.previous, 'squad_overlap_boxes', ()))
        return fresh, candidates

    def remember_manual_entry(self, stage, target=None):
        """需手动关卡关闭后记进设施目录；目录无法配准时跳过。

        点了画面内目标时按点击前那帧的脚圈记位置；没有点击目标（途中弹出）时才退回小队脚下。
        实录 2026-10-22（22-36）：点远处 EX 红圈后备战直接弹出、小队没走过去，四个不同 EX 都被
        记在小队脚下同一点，之后照样被选中。
        """
        from module.campaign.facility_memory import FacilityUnconfirmed
        from module.campaign.structures import facility_catalog
        if target is not None and self.tap_frame is not None:
            field, position, source = self.tap_frame, np.asarray(target['click'], float), '点击前的脚圈'
        else:
            field = self.map_frame(track=False)
            arrow = squad_arrow(field)
            if arrow is None:
                logger.info('关闭需手动关卡后未看到小队，未记下它的位置。')
                return
            position, source = np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET], '小队脚下'
        try:
            facility_catalog(self).remember_manual(field, position, stage)
        except (FacilityUnconfirmed, OSError) as exc:
            logger.info(f'未记下需手动关卡位置：{exc}')
            return
        logger.info(f'按{source} {np.round(position).astype(int).tolist()} 记下需手动关卡 {stage_text(stage)}，'
                    '之后选目标时跳过这里的敌人。')

    def manual_entry_check(self, target, field):
        """目标落点是否在已记下的需手动关卡旁；返回 'skip' 或 None（目录无法配准时照常尝试）。"""
        from module.campaign.facility_memory import FacilityUnconfirmed
        from module.campaign.structures import facility_catalog
        try:
            catalog = facility_catalog(self)
            if not catalog.has_manual_entries():
                return None
            catalog.observe(field, [])
        except (FacilityUnconfirmed, OSError):
            return None
        self.session.record_evidence('manual_entry_check', field, target=target, projected=catalog.manual_projected)
        for entry in catalog.manual_projected:
            distance = float(np.linalg.norm(np.asarray(entry['position'])-target['click']))
            if distance <= MANUAL_ENTRY_RADIUS:
                logger.info(f"目标 {target['id']} 在已记下的需手动关卡 {entry['facility_id']} 旁"
                            f"（{distance:.0f}px），跳过。")
                return 'skip'
        return None

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
                and item['status'] not in ('deferred', 'ex_skipped', 'unreachable', 'stalled', 'manual_required'))

    def return_lobby(self, field):
        """连续确认主线要求返回大厅后点真实任务箭头一次，后续由大厅独立核对任务推进。

        两帧及最终输入帧都须确认同章普通地图、完整任务文字和蓝色箭头；文字已确认但
        控件或身份不稳定时停止，不继续扫怪。点击本身不计作任务完成或整章清空。
        """
        if not return_lobby_quest(read_quest_text(field, self.owner.chapter_ocr)):
            return False
        expected = (self.owner.current_chapter, 'normal')
        point = quest_jump_point(field)
        if point is None or chapter_identity(field, self.owner.chapter_ocr) != expected:
            raise StoryVerificationError('返回大厅任务已出现，但地图身份或任务箭头未确认。',
                                         reason='story_return_unconfirmed')

        def validate(latest):
            """最新一帧必须仍为同一返回任务、地图身份和实际箭头，不沿用旧屏幕点位。"""
            current = read_quest_text(latest, self.owner.chapter_ocr)
            control = quest_jump_point(latest)
            accepted = (return_lobby_quest(current) and chapter_identity(latest, self.owner.chapter_ocr) == expected
                        and control is not None and np.linalg.norm(np.asarray(point)-control) <= 5.)
            self.session.record_evidence('story_return_guard', latest, accepted=bool(accepted), quest=current,
                                         control=control, arrays=dict(field=latest))
            return bool(accepted)

        self.session.pause(.35)
        fresh = self.map_frame()
        if not validate(fresh):
            raise StoryVerificationError('返回大厅任务未能连续确认，未发送点击。',
                                         reason='story_return_unconfirmed')
        self.session.begin_target('story_return_lobby')
        self.session.record_evidence('story_return_lobby', fresh, chapter=expected[0], control=point,
                                     arrays=dict(field=fresh))
        if not self.session.tap_verified_ui(point, validate):
            raise StoryVerificationError('返回大厅任务点击未发送，保留当前状态。',
                                         reason='story_return_unconfirmed')
        logger.info('已核验返回大厅任务并点击任务箭头；待大厅实际任务核对，不计作通关或清理完成。')
        return True

    def quest_target(self, before, field, candidates):
        """用稳定主线标记给已有真实脚圈排序，不从标记推算新的落点或授予关卡身份。

        同章未完成任务才启用。标记至少两帧位置相差不超过10px；只有场内标记正下方
        0.25～3倍图标高度、横差不超过0.75倍图标宽度的唯一已确认脚圈可优先。
        无唯一关联时只报告有稳定提示，供有界镜头露出；离屏/弱形态提示不能直接关联落点。
        """
        if (self.quest is None or self.quest[0] != self.owner.current_chapter
                or self.quest in self.owner.story_wins):
            return None, False
        previous = story_markers(before)
        markers = [item for item in story_markers(field) if sum(
            np.linalg.norm(np.asarray(item['position'])-old['position']) <= 10. for old in previous) == 1]
        linked = {}
        for marker in markers:
            if marker['kind'] != 'scene':
                continue
            left, top, right, bottom = marker['box']
            width, height = right-left, bottom-top
            point = np.asarray(marker['position'])
            matching = [item for item in candidates
                        if abs(item['click'][0]-point[0]) <= .75*width
                        and .25*height <= item['click'][1]-point[1] <= 3.*height]
            if len(matching) == 1:
                linked[matching[0]['id']] = matching[0]
        target = next(iter(linked.values())) if len(linked) == 1 else None
        return target, bool(markers)

    def guide_quest(self, field):
        """每个当前主线最多一次请求游戏定位任务，再从新帧扫描而不猜测敌人坐标。

        同一完整任务号只尝试一次；两帧任务身份和右箭头都须一致。定位之后旧相机关联作废，
        若途中遭遇其他怪，主循环仍核对实际备战编号；此操作不替代本章末尾的清怪巡检。
        """
        if (self.quest is None or self.quest[0] != self.owner.current_chapter
                or self.quest in self.guided_quests or self.quest in self.owner.story_wins):
            return False
        focused = self.focus_quest(field, 'quest_guide', on_confirm=lambda: self.guided_quests.add(self.quest))
        if not focused:
            return False
        if focused == 'entry':
            return True
        logger.info(f'已请求游戏定位主线任务 {stage_text(self.quest)}，下一步重新扫描全部普通／分支红圈。')
        return True

    def focus_quest(self, field, purpose, on_confirm=None):
        """点击两帧一致的主线任务箭头，让游戏把镜头移到当前任务目标；返回'entry'、True或False。

        任务号、箭头位置及章节身份在真正输入前再核一次；复核漏读只放弃本次，不终止推图。
        on_confirm在两帧确认后、发送点击前调用（任务定位据此只尝试一次）。镜头移动不计进展。
        """
        point = quest_jump_point(field)
        if point is None or self.quest is None:
            return False
        self.session.pause(.25)
        fresh = self.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            return 'entry'
        current = read_stage(fresh, self.owner.chapter_ocr, quest=True)
        confirmed = quest_jump_point(fresh)
        if current != self.quest or confirmed is None or np.linalg.norm(np.asarray(point) - confirmed) > 5.:
            return False
        self.session.begin_target(purpose)
        self.session.record_evidence(purpose, fresh, quest=self.quest, control=confirmed)
        self.memory.previous = None
        expected_quest = self.quest

        def validate(latest):
            """游戏任务定位在真正输入前仍须匹配同章任务和实际蓝箭头，不能沿用旧提示。"""
            current = read_stage(latest, self.owner.chapter_ocr, quest=True)
            control = quest_jump_point(latest)
            return (current == expected_quest and control is not None
                    and np.linalg.norm(np.asarray(confirmed)-control) <= 5.
                    and chapter_identity(latest, self.owner.chapter_ocr) == (self.owner.current_chapter, 'normal'))

        if on_confirm is not None:
            on_confirm()
        try:
            if not self.session.tap_verified_ui(confirmed, validate):
                return False
        except TargetTriggered:
            return 'entry'
        except UiControlChanged:
            # 实录 2026-10-21（22-21）：任务文字OCR分数贴着0.9门槛（0.899/0.901），输入前复核漏读；
            # 定位只是辅助，不点击并交回主循环，不能因此终止整轮推图。
            logger.info('主线任务控件在输入前复核未通过，未点击；交回主循环继续扫描。')
            return False
        self.session.pause(1.5)
        return True

    def reveal_marked_area(self, field, has_targets=False):
        """用稳定的场内／离屏菱形露出目标区域，不把屏幕标记当作地面落点或敌人身份。

        每次目标定位最多四次，设施动作后允许重新定位；真实胜利前设施总预算仍是两次。
        拖动不超过480px。场内／离屏分类会随指针动画变化，
        最多六帧内两次同位置识别才用于找镜头方向，不直接点击。小队精定位或离屏标记须明确，
        实际相机手势继续经过现有按下点避让；操作后必须重新扫描脚圈并核对真实备战。
        has_targets 表示当前画面另有可直接点击的普通脚圈：小队位置读不到时交回主循环先打它们。
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
        # 顶栏夹住的菱形实测都在 y≈225；场内菱形（如 21-32 的 y≈133～159）点了会触发原生寻路，不能当作边缘。
        clamped = 200. <= target[1] <= 250. or target[0] <= 100. or target[0] >= 1676.
        if self.marker_jumps < 2 and length > 250. and clamped:
            # 实录 2026-10-21（21-15）：目标在多屏之外时菱形贴在顶栏下沿（y≈225）或左右边缘，480px
            # 小步平移追不上；
            # 点一下边缘菱形，游戏会把镜头移到任务目标（同头像回中）。只移动镜头，之后重新扫描脚圈；
            # 场内菱形仍按原规则只平移镜头。
            self.marker_jumps += 1
            point = target.tolist()

            def validate(latest):
                """真正输入前最新帧在原位置仍有主线菱形；发光动画可能只剩白色内框（cue）。"""
                return any(np.linalg.norm(np.asarray(item['position'])-target) <= 15.
                           for item in story_markers(latest))

            self.session.begin_target('marker_jump')
            self.session.record_evidence('marker_jump', fresh, marker=marker, arrays=dict(field=fresh))
            try:
                tapped = self.session.tap_verified_ui(point, validate)
            except TargetTriggered:
                return True
            except UiControlChanged:
                # 实录 2026-10-21：右下边缘菱形闪烁，输入前复核恰好漏帧；不点击，下一轮重新观测。
                logger.info('任务菱形在输入前复核未通过，未点击；下一轮重新观测。')
                return True
            if tapped:
                self.memory.previous = None
                logger.info('已点击任务菱形，由游戏把镜头移到主线目标；重新扫描真实脚圈，不将菱形作为战斗落点。')
                self.session.pause(1.5)
                return True
        arrow = squad_arrow(fresh)
        if arrow is None and squad_offscreen(fresh) is None:
            if has_targets:
                # 实录 2026-10-21（21-32）：关掉残留备战弹窗后小队与普通敌人叠在一起，头顶标记不显示；
                # 身边两个敌人可直接点击，不必为首领进入无锚点的设施恢复（那里会因缺少小队锚点停止）。
                logger.info('小队位置未读到，但画面内有可点击的普通敌人，先交回主循环处理。')
                return False
            if self.quest is not None and self.quest not in self.owner.story_wins:
                from module.campaign.structures import recover_visible_structure
                self.session.record_evidence('marker_camera_unanchored', fresh, marker=marker,
                                             arrays=dict(field=fresh))
                recover_visible_structure(self, fresh, allow_unanchored=True)
                return True
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

    def follow_mainline(self, field, quest):
        """持续模式确认两帧下一章主线任务后切章；旧章待办保留，不生成清空证明。

        只接受紧邻下一章，最后输入帧仍须同一旧章及相同完整任务。点击一次后核对两帧
        新章NORMAL身份，加载和未知状态受原地图等待及会话时限约束。
        """
        expected = (self.owner.current_chapter, 'normal')
        if (not getattr(self.owner, 'continuous', False) or quest is None
                or quest[0] != expected[0] + 1):
            return None
        self.session.pause(.35)
        fresh = self.map_frame()
        if (chapter_identity(field, self.owner.chapter_ocr) != expected
                or chapter_identity(fresh, self.owner.chapter_ocr) != expected
                or read_stage(fresh, self.owner.chapter_ocr, quest=True) != quest):
            return None

        def valid(latest):
            """实际切章前仍核对旧章身份和完整新主线任务，不接受过时OCR结论。"""
            return (chapter_identity(latest, self.owner.chapter_ocr) == expected
                    and read_stage(latest, self.owner.chapter_ocr, quest=True) == quest)

        self.session.begin_target('mainline_next_chapter')
        self.session.record_evidence('mainline_next_chapter', fresh, quest=quest, old_chapter_pending=True)
        self.session.tap_verified_ui((1715, 940), valid)
        deadline, previous = time.monotonic() + 30., None
        while time.monotonic() < deadline:
            current = chapter_identity(self.map_frame(), self.owner.chapter_ocr)
            if current == previous and current == (quest[0], 'normal'):
                logger.info(f'主线任务与新章身份已连续确认：第 {quest[0]} 章；旧章可选内容仍待办。')
                return dict(state='continue', reason='chapter_changed', old_chapter_pending=True)
            previous = current
            self.session.pause(.35)
        raise StoryVerificationError('主线切章后未能连续确认新章身份，旧章待办保留。', 'story_chapter_unconfirmed')

    def run(self, ticks=48):
        """当前主线先复用游戏任务定位；持续模式跟随已确认的新章，单章清理保留巡检。

        关卡顺序只作为辅助信息：已知遇敌编号从小到大、未知目标按距离排序；每次开战仍独立
        核对实际编号。总时限、单目标输入预算、停止信号及首次战败终止继续由正式会话执行。
        """
        for _ in range(ticks):
            self.session.begin_target('scene_scan')
            field = self.map_frame()
            if goto.battle_popup_score(field) > .8:
                self.fight_entry()
                continue
            if not self.chapter_surveyed:
                self.chapter_surveyed = True
                from module.campaign.structures import chapter_start_survey
                surveyed = chapter_start_survey(self)
                if surveyed == 'entry':
                    self.fight_entry()
                if surveyed is not None:
                    continue
            identity = chapter_identity(field, self.owner.chapter_ocr)
            if identity is not None and identity[0] > self.owner.current_chapter:
                if getattr(self.owner, 'continuous', False):
                    self.session.pause(.35)
                    fresh = self.map_frame()
                    if (identity[1] == 'normal'
                            and chapter_identity(fresh, self.owner.chapter_ocr) == identity):
                        return dict(state='continue', reason='chapter_changed', old_chapter_pending=True)
                    raise StoryVerificationError('自动换章后身份无法连续确认，旧章待办保留。',
                                                 'story_chapter_unconfirmed')
                self.restore_chapter(identity)
                continue
            current = read_stage(field, self.owner.chapter_ocr, quest=True)
            if current is None and self.return_lobby(field):
                return dict(state='continue', reason='story_return_lobby', chapter=self.owner.current_chapter)
            if current is not None:
                if current != self.quest:
                    self.memory.reconsider(current, self.owner.story_wins)
                    self.deferred.clear()
                self.quest = current
                advanced = self.follow_mainline(field, current)
                if advanced is not None:
                    return advanced
                if self.guide_quest(field):
                    continue
            fresh, candidates = self.scan(field)
            if goto.battle_popup_score(fresh) > .8:
                self.fight_entry()
                continue
            candidates = [item for item in candidates if self.selectable(item)]
            preferred, marked = self.quest_target(field, fresh, candidates)
            if marked and preferred is None and self.reveal_marked_area(fresh, has_targets=bool(candidates)):
                continue
            # 实录 2026-10-21（22-27）：受阻时画面内还有 3 个可选敌人，挡在机关前的支线怪就在其中，
            # 原先直接进入设施恢复，一个都没打。恢复开始后则保持恢复优先，不插入敌人点击。
            if self.blocked_views and preferred is None and not (self.try_visible and candidates):
                self.try_visible = False
                from module.campaign.structures import recover_visible_structure
                recovery = recover_visible_structure(self, fresh)
                if recovery != 'scouted':
                    self.blocked_views = 0
                continue
            if not candidates:
                if self.guide_quest(fresh):
                    continue
                if not marked and self.reveal_marked_area(fresh):
                    continue
                survey = ChapterSweep(self).run()
                if survey['state'] == 'entry':
                    self.fight_entry()
                    continue
                if survey['state'] == 'verified_empty':
                    return self.advance_chapter(survey['proof'])
                fresh, candidates = survey['field'], survey['targets']
            observed_arrow = squad_arrow(fresh)
            anchor = np.asarray(goto.VIEWPORT_PIXEL if observed_arrow is None else observed_arrow)
            target = preferred or min(candidates, key=lambda item: (
                stage_order(item['stage']) if item['stage'] is not None else (99, 99, '', 0),
                float(np.linalg.norm(np.asarray(item['click']) - anchor))))
            # 每个目标编号只核对一次，避免同一目标每轮都重新配准目录。
            checked = None if target['id'] in self.manual_checked else self.manual_entry_check(target, fresh)
            self.manual_checked.add(target['id'])
            if checked == 'skip':
                self.deferred.add(target['id'])
                continue
            self.session.begin_target(target['id'])
            self.session.preview(fresh)
            self.session.record_evidence('scene_target', fresh, quest=self.quest, target=target, arrays=dict(field=fresh))
            logger.info(f"地图敌人目标：{target['id']}，脚圈 {target['click']}，当前可选 {len(candidates)} 个；"
                        '到达后读取实际遇敌关卡。')
            self.tap_frame = fresh
            try:
                self.session.tap(target['click'])
            except TargetTriggered:
                # 最后一道输入闸门可能比扫描更早读到途中弹窗；只核对实际备战，不补发旧落点。
                pending = self.map_frame()
                if goto.battle_popup_score(pending) > .8:
                    self.fight_entry()
                continue
            if self.wait_entry(fresh, arrow=observed_arrow, target=target):
                won = self.fight_entry(target)
                if not won:
                    self.deferred.add(target['id'])
                elif self.blocked_views:
                    # 打掉挡路的怪后道路可能已通：重新尝试暂缓的目标，而不是继续设施恢复。
                    logger.info('受阻期间打赢一个可见敌人，重新尝试之前受阻的目标。')
                    self.blocked_views, self.try_visible = 0, False
                    self.deferred.clear()
            else:
                self.memory.attempt(target['id'], 'stalled', self.entry_outcome)
                self.blocked_click = list(target['click'])
                self.blocked_views += 1
                self.try_visible = True
                logger.info('原生寻路已确认连续停滞：先试画面内其余可见敌人，都走不到再进入有界设施诊断。')
                self.deferred.add(target['id'])
        return dict(state='continue', reason='tick_budget')


def run_visible_story(owner):
    """在加载底图前运行清怪闭环；所有退出路径释放会话，未完成巡检不得声明本章清空。"""
    from module.campaign.execution import CampaignExecutor
    from module.campaign.session import CampaignSession
    from module.campaign.vision.detection import configure_inference
    owner.story_wins = set()
    owner.prepare_client()
    owner.check_main_story_page()
    owner.detect_current_chapter(localize=False)
    session = CampaignSession(owner, navigation=False)
    inference_worker = None
    try:
        inference_worker = configure_inference(owner.config.config_name)
        session.start()
        return VisibleStoryRunner(owner, session, CampaignExecutor(owner, session)).run()
    finally:
        try:
            session.close()
        finally:
            from module.campaign.vision.inference_backend import stop_worker
            stop_worker(inference_worker)
