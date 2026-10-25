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
from module.campaign.rejected_targets import RejectedTargets, static_view
from module.campaign.vision.perception import squad_arrow, squad_ground
from module.campaign.vision.offscreen import squad_offscreen
from module.campaign.vision.story_marker import story_markers
from module.campaign.vision.native_motion import ClickProgress, MinimapStability, NativeMotion, combine_motion
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.window import UiControlChanged
from module.logger import logger

# 目标脚圈离已记下的需手动关卡（小队触发备战时的脚下位置）多近算同一个敌人（客户区像素）：
# 小队在脚圈约28底图像素内触发备战，约75客户区像素，再留关键帧投影误差。
MANUAL_ENTRY_RADIUS = 120.
# 受阻时脚圈离目录里仍关闭的地面机关多近算「守机关的怪」，优先打（客户区像素）。
# 实录 2026-10-24（25章）：机关上那只怪的脚圈离机关中心约 23px，再留关键帧投影误差。
SWITCH_GUARD_RADIUS = 90.
# 判定停滞后再等多久补看一帧：小队可能只是途中短暂停顿，备战随后弹出。
STALL_FOLLOWUP_SECONDS = 2.

QUEST_AREA = (1565, 88, 1710, 114)
# 光幕作受阻证据的检测分门槛（只用于转入找机关流程，不作点击）。
BARRIER_EVIDENCE = .8
# 「前往」任务行的识别分门槛：实录 2026-10-23「前往【方舟】的【市内】」在地图上只有 0.868～0.892。
DESTINATION_SCORE = .8
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
    困难难度标题在编号和后缀之间多一个 HARD（实录 2026-10-23：「4-1 HARD STAGE」）。
    """
    number = r'[1-9]\d?'
    separator = r'[-－–]'
    identity = rf'{number}\s*{separator}\s*{number}(?:\s*[A-Z]\s*{separator}\s*{number})?'
    match = re.fullmatch(rf'\s*({identity})\s*(?:HARD\s*)?(?:STAGE|B[O0]SS)\s*', text, re.I)
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


def read_quest_text(field, reader, minimum=.9):
    """读取固定主线任务行，默认0.9分门槛；兼容1776×999地图和720×1280大厅。

    返回高分原文或None；任务类型由调用方独立解析，不从当前章节或已胜关卡补全。
    minimum 只供「前往」任务用 DESTINATION_SCORE：长中文行放大后分数偏低，调用方须另行
    要求整行格式完全匹配且连续两帧文字相同。
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
    return text if np.isfinite(score) and score >= minimum else None


def return_lobby_quest(text):
    """只接受完整的简繁体返回大厅任务，不从含大厅的说明句或残缺文字猜测操作。"""
    return bool(text is not None and re.fullmatch(r'返回(?:【(?:大厅|大廳)】|\[(?:大厅|大廳)\]|大厅|大廳)',
                                                re.sub(r'\s+', '', text)))


def destination_lobby_quest(text):
    """大厅「前往【某处】」任务：通关后游戏要求去某个界面触发剧情或解锁，点任务箭头由游戏带过去。

    实录 2026-10-21：21 章通关后为「前往【前哨基地】」，进入基地即完成。只接受整行“前往＋括号地名”，
    「前往地面【出击】」是出击指令，不在此列。
    实录 2026-10-23：23 章通关后为「前往【方舟】的【市内】」，之后是「前往【方舟】的【简报室】」，
    接受“前往【地点】的【地名】”两段式；括号可能被 OCR 读成半角，左右各自兼容。
    """
    return bool(text is not None and re.fullmatch(
        r'前往(?:[【\[][^】\]]{1,12}[】\]](?:的[【\[][^】\]]{1,12}[】\]])?|前哨基地)', re.sub(r'\s+', '', text)))


def destination_place(text):
    """前往任务最终要点进的地名：「前往【方舟】的【市内】」为市内；不是前往任务返回None。"""
    if not destination_lobby_quest(text):
        return None
    names = re.findall(r'[【\[]([^】\]]{1,12})[】\]]', re.sub(r'\s+', '', text))
    return names[-1] if names else '前哨基地'


def read_ex_entry(field, reader):
    """EX 备战同时核对独立标题模板与完整 OCR 标题，不凭共有 STAGE 字样认领 EX。

    EX 标题没有普通数字关卡身份，调用方只记录当前章及内部遇敌序号，不虚构游戏编号。
    """
    if not battle.ex_stage(field):
        return False
    x1, y1, x2, y2 = ENTRY_AREA
    crop = cv2.resize(field[y1:y2, x1:x2], None, fx=2, fy=2)
    text = reader.ocr(crop, direct_ocr=True, threshold=.9)['text']
    # 困难难度为「EX HARD STAGE」（实录 2026-10-23 困难 4 章，300 钻、等级固定 190、无法自主战斗）
    return bool(re.search(r'(?<![A-Z])EX\s*(?:HARD\s*)?STAGE(?![A-Z])', text, re.I))


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
        # 点过却走不到／点了没反应的位置，按场景配准跟踪；脚圈换了编号也不再点同一处。
        self.rejected = RejectedTargets()
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
        # 任务已在下一章但切章核对未通过的轮数。
        self.next_chapter_misses = 0
        # 最近一次点击画面内目标时的那帧；需手动关卡按它记位置。
        self.tap_frame = None
        self.map_localizer = None
        # 受阻目标 → 已做螺旋机关搜索次数；已点过但未解决的搜索键（同状态下不重复点击）。
        self.switch_searches, self.switch_skip = {}, set()
        # 已用任务定位看过目标旁的受阻目标（每个只看一次）。
        self.goal_views = set()
        # 已因「任务定位处有光幕却没有可点敌人」进入找机关流程的主线任务（每个任务一次）。
        self.barrier_quests = set()
        self.marker_walks = set()
        # 刚受阻、设施恢复尚未开始：先把画面内其余可见敌人逐个试一遍（可能就是挡路的支线怪）。
        self.try_visible = False
        self.gate_enemy_tries = {}
        # 受阻目标 → 螺旋下次续看的屏号（被守关敌人打断后不从第0屏重来）；踩亮开关后清空。
        self.spiral_resume = {}
        # 已连续看完全部螺旋屏幕找过机关的受阻目标；踩亮开关后清空。
        self.spiral_done = set()
        self.blocked_since, self.blocked_click = {}, None
        self.entry_outcome = None
        # 困难难度与普通共用底图但敌人和通关进度独立，清怪记录分开存，不能互相顶替。
        self.difficulty = getattr(owner, 'current_difficulty', None) or 'normal'
        self.hard = self.difficulty == 'hard'
        self.journal = ClearanceJournal(f'{owner.config.config_name}-hard' if self.hard else owner.config.config_name)
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

    def wait_entry(self, before, arrow=None, target=None, recenter=True):
        """观察游戏原生寻路；只有真实移动才续等，停滞与观测失败分开处理。

        三次且跨2秒的可靠静止才返回False请求设施诊断。观测不足或持续移动超过总时限
        均保留未确认并停止，不能将二十秒没弹窗写成道路不可达，也不在途中追加目标点击。
        一直看不到小队但画面上有离屏头像时，点头像找回镜头后重新观察一次；仍看不到才停止。
        实录 2026-10-23（23-11）：点画面上沿的敌人后小队走出画面，小地图也没有圆环，
        20秒只得到 squad_unobserved 就中止了整轮，离屏头像其实一直在画面上。
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
            if (recenter and state == 'observation_unknown' and observation.get('reason') == 'squad_unobserved'
                    and squad_offscreen(field) is not None):
                from module.campaign.structures import StructureUnconfirmed, recenter_squad
                logger.info('原生寻路途中小队离开画面：点离屏头像找回镜头后重新观察一次。')
                try:
                    recenter_squad(self)
                except StructureUnconfirmed as exc:
                    raise StoryVerificationError(str(exc), 'native_recenter_failed')
                fresh = self.map_frame()
                # 镜头已换，点击前目标的画面坐标不再对应；只按新镜头观察小队。
                return self.wait_entry(fresh, arrow=squad_arrow(fresh), recenter=False)
            # 实录 2026-10-24（25章）：小队站在橙色圈上、头顶无箭头，小地图圆环也没有，把自己脚下的圈当敌人点；
            # 二十秒画面纹丝不动后整轮中止。镜头没动、没弹备战，说明这一下没点到能走过去打的敌人。
            if (state == 'observation_unknown' and target is not None
                    and observation.get('reason') == 'squad_unobserved' and static_view(before, field)):
                logger.info('点击后画面没变、小队和小地图圆环都看不到：当作点到的不是敌人（可能是小队自己脚下的圈），'
                            '记下位置换别的目标。')
                self.entry_outcome = dict(self.entry_outcome, no_effect=True)
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
        if goto.entry_locked(self.session.win.client()):
            # 实录 2026-10-23：任务 23-26 未完成时路上碰到 23-27，进入战斗被锁，快速战斗也被拒绝。
            # 按 deferred 记下，任务推进到该编号后观察簿自动放回候选。
            self.memory.identify(identifier, stage, 'deferred')
            self.close_entry()
            logger.info(f'已关闭备战 {stage_text(stage)}：进入战斗被锁（前置主线未完成），先打当前任务，继续扫描其他敌人。')
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

    def closed_switch_points(self, field):
        """目录里仍关闭的地面机关投影到本帧的位置；只用于给目标排序，不授予点击机关。

        目录无法配准时返回空，照常按离小队远近选目标。
        """
        from module.campaign.facility_memory import FacilityUnconfirmed
        from module.campaign.structures import facility_catalog
        try:
            catalog = facility_catalog(self)
            catalog.observe(field, [])
        except (FacilityUnconfirmed, OSError, cv2.error, np.linalg.LinAlgError):
            return []
        points = [np.asarray(item['position'], float) for item in catalog.expected]
        self.session.record_evidence('switch_guard_check', field, switches=[point.tolist() for point in points])
        return points

    def stall_followup(self, target):
        """判定停滞后隔一会儿补看一帧并留证；备战已弹出说明小队其实走到了，照常交战斗核对。

        实录 2026-10-24（25章）：机关上的怪被判停滞而放弃，人工点同一只怪能走到，但当时的帧已被
        32 槽覆盖，无法区分真走不到、小队跟丢还是途中短停。补看帧只认备战弹窗，不据此续等或补点。
        """
        self.session.pause(STALL_FOLLOWUP_SECONDS)
        field = self.map_frame()
        popup = goto.battle_popup_score(field) > .8
        self.session.record_evidence('native_stall_followup', field, target=target, feedback=self.entry_outcome,
                                     popup=popup, arrays=dict(field=field))
        if popup:
            logger.info(f"停滞判定后补看：{target['id']} 的备战已弹出，小队其实走到了，照常核对开战。")
            self.entry_outcome = dict(state='entry')
        return popup

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
                and item['status'] not in ('deferred', 'ex_skipped', 'unreachable', 'stalled', 'no_effect',
                                           'manual_required'))

    def drop_rejected(self, field, candidates):
        """去掉落在之前走不到／点了没反应位置上的候选；没有记录时不做配准。"""
        kept, skipped = self.rejected.filter(field, candidates)
        for item, distance, reason in skipped:
            text = '点了没反应' if reason == 'no_effect' else '走不到'
            logger.info(f"跳过 {item['id']}：离之前{text}的位置 {distance:.0f}px。")
        if skipped:
            self.session.record_evidence('rejected_skip', field, skipped=[
                dict(id=item['id'], click=item['click'], distance=round(distance, 1), reason=reason)
                for item, distance, reason in skipped])
        return kept

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

    def quest_barrier_blocked(self, field):
        """主线已定位、画面没有可点敌人但连续两帧有亮着的光幕时，按受阻进入找机关流程。

        清怪巡检只找敌人红圈，不看机关；目标被光幕挡住时扫多少屏也打不通。这里不点任何东西：
        只把小队找回画面并登记受阻，下一轮由 recover_visible_structure 先看目标旁、再螺旋和全章巡检找机关。
        每个主线任务只转一次，之后仍无进展按原流程巡检或停止。
        实录 2026-10-23（24-27）：目标在光幕后，任务定位后一个红圈都没有，转去清怪巡检扫了 32 屏，
        机关不在敌人巡检的识别范围内，最后镜头被拖出地图停止。
        """
        if (self.quest is None or self.quest[0] != self.owner.current_chapter
                or self.quest not in self.guided_quests or self.quest in self.owner.story_wins
                or self.quest in self.barrier_quests):
            return False
        from module.campaign.vision.objects import detect_objects

        def barriers(frame):
            """亮着的光幕（模型状态 curtain；关掉后只剩 pillar）；只作受阻证据，不作点击目标。"""
            return [item for item in detect_objects(frame, ('barrier',))
                    if item['state'] == 'curtain' and item['confidence'] >= BARRIER_EVIDENCE]

        first = barriers(field)
        if not first:
            return False
        self.session.pause(.35)
        fresh = self.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            self.fight_entry()
            return True
        second = barriers(fresh)
        if not second:
            return False
        self.barrier_quests.add(self.quest)
        self.session.record_evidence('quest_barrier_blocked', fresh, quest=self.quest, barriers=second,
                                     arrays=dict(field=fresh))
        logger.info(f'主线 {stage_text(self.quest)} 定位处有光幕且没有可点敌人：按受阻进入找机关流程，'
                    '不做只找敌人的清怪巡检。')
        from module.campaign.structures import recenter_squad
        recenter_squad(self)
        self.blocked_click = None
        self.blocked_views += 1
        self.try_visible = False
        return True

    def walk_to_marker(self, field):
        """画面内两帧同位置的主线菱形下认不出红圈时，点菱形让游戏原生寻路走到下一关。

        用户确认（2026-10-23）：画面正前方的「!」菱形就是下一个敌人。实录 24-37：Boss 入口是带骷髅的
        红色金字塔，光圈被遮住一半，入口和 Boss 识别都没过，转去清怪巡检空转后停止。点场内菱形会
        触发原生寻路（见 21-32 实录）。每个主线任务只走一次；开战前仍由备战弹窗核对实际关卡。
        """
        if (self.quest is None or self.quest[0] != self.owner.current_chapter
                or self.quest in self.owner.story_wins or self.quest in self.marker_walks):
            return False
        # 菱形发光动画让单帧分数在门槛附近起伏（实录 24-37：6帧里只有2帧过线），与 reveal_marked_area
        # 一样最多看6帧，两帧同位置才算确认。
        previous, marker, fresh = [], None, field
        for index in range(6):
            if index:
                self.session.pause(.25)
                fresh = self.map_frame()
                if goto.battle_popup_score(fresh) > .8:
                    self.fight_entry()
                    return True
            markers = [item for item in story_markers(fresh) if item['kind'] == 'scene']
            confirmed = [item for item in markers if any(
                np.linalg.norm(np.asarray(item['position'])-old['position']) <= 10. for old in previous)]
            if len(confirmed) == 1:
                marker = confirmed[0]
                break
            if markers:
                previous = markers
        if marker is None:
            return False
        target = dict(id=f'quest-marker-{stage_text(self.quest)}', click=list(marker['position']), stage=None)
        self.marker_walks.add(self.quest)
        self.session.begin_target(target['id'])
        self.session.record_evidence('marker_walk', fresh, quest=self.quest, marker=marker, arrays=dict(field=fresh))
        logger.info(f"主线 {stage_text(self.quest)} 菱形 {target['click']}（{marker['confidence']:.2f}）下未认出红圈："
                    '点菱形交由原生寻路前往，开战前核对实际备战关卡。')
        try:
            self.session.tap(target['click'])
        except TargetTriggered:
            if goto.battle_popup_score(self.map_frame()) > .8:
                self.fight_entry()
            return True
        if self.wait_entry(fresh, arrow=squad_arrow(fresh), target=target):
            self.fight_entry()
        else:
            logger.info('点主线菱形后原生寻路停滞，交回主循环按受阻流程处理。')
        self.memory.previous = None
        return True

    def destination_return(self, field):
        """地图任务为「前往【某处】」时交回大厅流程，由大厅点任务箭头前往；地图上不点击。

        长中文行分数偏低，按 DESTINATION_SCORE 读取，但须整行格式匹配且相隔一帧文字完全相同。
        实录 2026-10-23（23 章通关）：任务变为「前往【方舟】的【市内】」，地图读不出关卡号，
        转去旧章全章巡检，最后在地图边缘因斜向镜头响应停止。
        """
        text = read_quest_text(field, self.owner.chapter_ocr, minimum=DESTINATION_SCORE)
        if not destination_lobby_quest(text):
            return False
        self.session.pause(.35)
        again = read_quest_text(self.map_frame(), self.owner.chapter_ocr, minimum=DESTINATION_SCORE)
        if again != text:
            return False
        logger.info(f'主线任务为 {text!r}：交回大厅点任务箭头前往，旧章可选内容保留待办。')
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
        """登记当前主线已在本图核对过，不发送任何输入；始终返回False交回扫描。

        任务栏右侧的蓝色箭头只用于跳转到其他功能模块，局内点了镜头不动、也不引路。
        实录 2026-10-23：历史34次局内点击前后帧比对，镜头一次都没有移动（差异只是场景动画）；
        24-34 时还因此误以为任务目标已在画面里，转去清怪巡检。用户确认该箭头局内无用。
        guided_quests 仍作为「本任务已核对过当前画面」的标记供光幕受阻判断使用。
        """
        if (self.quest is not None and self.quest[0] == self.owner.current_chapter
                and self.quest not in self.owner.story_wins):
            self.guided_quests.add(self.quest)
        return False

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
            observed = story_markers(fresh, top_bar=True)
            observations.append(observed)
            markers = [item for item in observed if item['kind'] in ('scene', 'edge', 'cue', 'hud')]
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
        # 顶栏里的菱形只给方向：点那里会点到顶栏按钮，只平移镜头。
        if self.marker_jumps < 2 and length > 250. and clamped and marker['kind'] != 'hud':
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
        ground, _ = squad_ground(fresh)
        if ground is None and squad_offscreen(fresh) is None:
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
            if current[1] != self.difficulty:
                raise SweepUnconfirmed('换章后难度变化，不能复用旧章清怪状态。')
            self.session.tap((1551, 940), interval=2.)
            self.session.pause(1.)
            field = self.map_frame()
            observed = chapter_identity(field, self.owner.chapter_ocr)
            if observed is not None:
                current = observed
        if current != (self.owner.current_chapter, self.difficulty):
            raise SweepUnconfirmed('未能返回尚未完成巡检的章节，清怪状态保持未完成。')
        self.memory.previous = None
        self.deferred.clear()
        self.rejected.clear()

    def advance_chapter(self, proof):
        """本章完整巡检无剩余敌人且任务已进入下一章后，才点击下一章节并核对新身份。"""
        # 困难难度没有自己的主线任务（任务栏显示的是普通主线），只凭本章完整巡检判断。
        if proof.get('state') != 'verified_empty' or not self.hard and (
                self.quest is None or self.quest[0] <= self.owner.current_chapter):
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
            if (current is not None and current == previous
                    and current == (self.owner.current_chapter + 1, self.difficulty)):
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
        # 实录 2026-10-22（22→23）：任务已是「通关【23-3】」，复核帧有一项没过就静默返回，
        # 主循环随后进了旧章全章巡检空转。复核最多重读三帧，并记录每次没过的是哪一项。
        previous = field
        for attempt in range(3):
            self.session.pause(.35)
            fresh = self.map_frame()
            checks = dict(old_identity=chapter_identity(previous, self.owner.chapter_ocr),
                          fresh_identity=chapter_identity(fresh, self.owner.chapter_ocr),
                          quest=read_stage(fresh, self.owner.chapter_ocr, quest=True))
            if checks['old_identity'] == checks['fresh_identity'] == expected and checks['quest'] == quest:
                break
            logger.info(f'切到第 {quest[0]} 章前复核未通过（第 {attempt+1} 次）：章节 {checks["old_identity"]}／'
                        f'{checks["fresh_identity"]}，任务 {checks["quest"]}。')
            previous = fresh
        else:
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
                    if (identity[1] == self.difficulty
                            and chapter_identity(fresh, self.owner.chapter_ocr) == identity):
                        return dict(state='continue', reason='chapter_changed', old_chapter_pending=True)
                    raise StoryVerificationError('自动换章后身份无法连续确认，旧章待办保留。',
                                                 'story_chapter_unconfirmed')
                self.restore_chapter(identity)
                continue
            # 困难地图的任务栏仍是普通主线任务：不跟随、不返回大厅，只清本章可见敌人并巡检。
            current = None if self.hard else read_stage(field, self.owner.chapter_ocr, quest=True)
            if current is None and not self.hard and (self.return_lobby(field) or self.destination_return(field)):
                return dict(state='continue', reason='story_return_lobby', chapter=self.owner.current_chapter)
            if current is not None:
                if current != self.quest:
                    self.memory.reconsider(current, self.owner.story_wins)
                    self.deferred.clear()
                self.quest = current
                advanced = self.follow_mainline(field, current)
                if advanced is not None:
                    return advanced
                if getattr(self.owner, 'continuous', False) and current[0] == self.owner.current_chapter+1:
                    # 任务已在下一章：旧章可选内容按设计保留待办，不在旧章扫描或巡检。
                    self.next_chapter_misses += 1
                    if self.next_chapter_misses >= 3:
                        raise StoryVerificationError(
                            f'主线任务已在第 {current[0]} 章，切章前核对连续三轮未通过；未在旧章巡检，已停止。',
                            'story_chapter_unconfirmed')
                    continue
                if self.guide_quest(field):
                    continue
            fresh, candidates = self.scan(field)
            if goto.battle_popup_score(fresh) > .8:
                self.fight_entry()
                continue
            candidates = self.drop_rejected(fresh, [item for item in candidates if self.selectable(item)])
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
                if self.quest_barrier_blocked(fresh):
                    continue
                if self.walk_to_marker(fresh):
                    continue
                try:
                    survey = ChapterSweep(self).run()
                except SweepUnconfirmed as exc:
                    if self.quest is None or self.quest[0] != self.owner.current_chapter:
                        raise
                    # 实录 2026-10-23（24-34）：目标在传送器另一侧，传送器是建筑上旋转的箭头，暂不识别；
                    # 巡检把镜头拖进云层后配准失败。停止原因要让人知道去看地图机关，而不只是配准失败。
                    raise SweepUnconfirmed(f'{exc}；主线 {stage_text(self.quest)} 当前画面未识别到敌人和任务标记，'
                                           '可能需要踩传送器、或入口样式脚本尚不认识，请人工查看处理后再继续') from exc
                if survey['state'] == 'entry':
                    self.fight_entry()
                    continue
                if survey['state'] == 'verified_empty':
                    return self.advance_chapter(survey['proof'])
                fresh, candidates = survey['field'], self.drop_rejected(survey['field'], survey['targets'])
                if not candidates:
                    raise StoryVerificationError('巡检找到的目标都在之前走不到或点了没反应的位置上，未再点击；'
                                                 '请人工查看是否需要机关或手动处理。', 'story_rejected_only')
            observed_arrow = squad_arrow(fresh)
            anchor = np.asarray(goto.VIEWPORT_PIXEL if observed_arrow is None else observed_arrow)
            # 受阻后先打守在关闭机关上／旁边的怪：打掉它小队就站在机关旁，路可能随之打开。
            # 实录 2026-10-24（25章）：按离小队远近先点了右边几只走不到的怪，机关上的怪排在后面。
            switches = self.closed_switch_points(fresh) if preferred is None and self.try_visible else []

            def switch_distance(item):
                return min((float(np.linalg.norm(np.asarray(item['click'])-point)) for point in switches),
                           default=float('inf'))
            target = preferred or min(candidates, key=lambda item: (
                switch_distance(item) > SWITCH_GUARD_RADIUS,
                stage_order(item['stage']) if item['stage'] is not None else (99, 99, '', 0),
                float(np.linalg.norm(np.asarray(item['click']) - anchor))))
            if target is not preferred and switch_distance(target) <= SWITCH_GUARD_RADIUS:
                logger.info(f"受阻期间优先打守机关的敌人：{target['id']} 离关闭机关 {switch_distance(target):.0f}px。")
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
            entered = self.wait_entry(fresh, arrow=observed_arrow, target=target)
            if not entered:
                entered = self.stall_followup(target)
            if entered:
                won = self.fight_entry(target)
                if not won:
                    self.deferred.add(target['id'])
                elif self.blocked_views:
                    # 打掉挡路的怪后道路可能已通：重新尝试暂缓的目标，而不是继续设施恢复。
                    logger.info('受阻期间打赢一个可见敌人，重新尝试之前受阻的目标。')
                    self.blocked_views, self.try_visible = 0, False
                    self.deferred.clear()
                    self.rejected.clear()
            else:
                no_effect = bool(self.entry_outcome and self.entry_outcome.get('no_effect'))
                reason = 'no_effect' if no_effect else 'stalled'
                self.rejected.add(fresh, target['click'], reason)
                self.memory.attempt(target['id'], reason, self.entry_outcome)
                if no_effect:
                    # 点到的不是能走过去的东西，不算道路受阻，不进入设施诊断。
                    self.deferred.add(target['id'])
                    continue
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
