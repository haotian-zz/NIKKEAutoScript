"""主线直接闭环的输入边界回归；隔离定义加载，不依赖 Windows、OCR 模型或真实游戏。"""

import ast
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from module.campaign.navigation.evidence import ActionEvidence
from module.campaign.vision.story_marker import story_markers


def definitions():
    """加载生产文件的定义并注入图像、时间与识别替身；测试不触碰窗口或输入驱动。"""
    path = Path(__file__).resolve().parents[1] / 'module/campaign/visible_story.py'
    tree = ast.parse(path.read_text('utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
    namespace = dict(TargetTriggered=type('TargetTriggered', (Exception,), {}), re=re, cv2=cv2, np=np,
                     time=SimpleNamespace(monotonic=Mock(return_value=0.)), logger=Mock(),
                     chapter_identity=Mock(return_value=(20, 'normal')),
                     goto=SimpleNamespace(battle_popup_score=Mock(return_value=0.), entry_locked=Mock(return_value=False), VIEWPORT_PIXEL=(888., 500.)),
                     battle=Mock(), ClearanceJournal=Mock(),
                     ChapterOverlayFlow=Mock(return_value=SimpleNamespace(handle=Mock(return_value=False))),
                     SceneEnemyMemory=Mock(),
                     stable_candidates=Mock(),
                     SweepUnconfirmed=type('SweepUnconfirmed', (RuntimeError,),
                                           {'reason': 'chapter_clearance_unconfirmed'}),
                     squad_arrow=Mock(return_value=None), squad_offscreen=Mock(return_value=None),
                     story_markers=story_markers, pan_scene=Mock(),
                     UiControlChanged=type('UiControlChanged', (RuntimeError,), {}))
    from module.campaign.rejected_targets import RejectedTargets
    from module.campaign.world import NEVER, WorldModel
    namespace.update(RejectedTargets=RejectedTargets, static_view=Mock(return_value=False),
                     WorldModel=WorldModel, NEVER=NEVER)
    namespace.update(read_title=Mock(return_value=None), title_crop=Mock(return_value=None))
    from module.campaign.vision.native_motion import ClickProgress, NativeMotion, combine_motion
    namespace.update(ClickProgress=ClickProgress, NativeMotion=NativeMotion, combine_motion=combine_motion,
                     MinimapStability=Mock(return_value=SimpleNamespace(observe=lambda field: dict(state='unknown'))))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)

    def squad_ground(image):
        """与生产相同：箭头优先；测试替身不读小地图。"""
        arrow = namespace['squad_arrow'](image)
        if arrow is None:
            return None, None
        return np.asarray(arrow, float)+[0., getattr(namespace['goto'], 'ARROW_GROUND_OFFSET', 0.)], 'arrow'
    namespace['squad_ground'] = squad_ground
    return namespace


class VisibleStoryTests(unittest.TestCase):
    """只有唯一目标和独立确认的关卡才允许继续，不把胜利等同于主线已前进。"""

    def setUp(self):
        """各场景从同一静止截图、20-6 主线与假会话开始，所有点击只写入 Mock。"""
        self.ns = definitions()
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.target = dict(id='ring-1', click=[600., 700.], stage=None, status='pending')
        self.session = Mock()
        self.session.evidence.folder = Path('/unused/mock-evidence')
        self.session.win.client.return_value = self.field
        self.owner = SimpleNamespace(current_chapter=20, chapter_ocr=Mock(), story_wins=set(), device=Mock(),
                                     config=SimpleNamespace(config_name='test'), continuous=True)
        self.executor = Mock()
        self.runner = self.ns['VisibleStoryRunner'](self.owner, self.session, self.executor)
        self.runner.map_frame = Mock(return_value=self.field)
        # 没有可点敌人时的探索单独验证（test_campaign_explorer）；这里默认探索看完什么都没找到。
        self.runner.explore = Mock(return_value=dict(state='exhausted', views=3, pans=2, duration=20.))
        # 需手动关卡的底图记忆单独验证（test_campaign_facility_map）。
        self.runner.remember_manual_entry = Mock()
        self.runner.manual_entry_check = Mock(return_value=None)
        # 受阻排序用的机关目录单独验证（test_closed_switch_points_projects_catalog）。
        self.real_closed_switch_points = self.runner.closed_switch_points
        self.runner.closed_switch_points = Mock(return_value=[])
        # 没有敌人时转找已知机关单独验证（test_known_closed_switch_enters_recovery_before_sweep）。
        self.real_known_switch_blocked = self.runner.known_switch_blocked
        self.runner.known_switch_blocked = Mock(return_value=False)
        self.reveal = Mock()
        patcher = patch.dict(sys.modules, {'module.campaign.navigation.story_navigation':
                                         SimpleNamespace(reveal_story_marker=self.reveal)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_manual_entry_check_runs_once_per_target_and_skips_known_ex(self):
        """已记下的需手动关卡旁的目标跳过；其他目标每个编号只核对一次。"""
        self.ns['read_stage'] = Mock(return_value=(20, 9))
        self.runner.manual_entry_check = Mock(side_effect=['skip', None])
        other = dict(self.target, id='ring-2', click=[1500., 800.])
        self.runner.scan = Mock(return_value=(self.field, [self.target, other]))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.inspect_entry = Mock(return_value=(20, 7))
        self.runner.run(ticks=2)
        self.assertEqual([call.args[0]['id'] for call in self.runner.manual_entry_check.call_args_list],
                         ['ring-1', 'ring-2'])
        self.session.tap.assert_called_once_with([1500., 800.])

    def test_manual_entry_recorded_at_tapped_ring_not_squad(self):
        """实录22-36：点远处 EX 后备战直接弹出、小队没走过去；按点击前那帧的脚圈记位置，途中弹出才用小队脚下。"""
        runner = self.ns['VisibleStoryRunner'](self.owner, self.session, self.executor)
        runner.map_frame = Mock(return_value=self.field)
        catalog = Mock()
        before = np.ones_like(self.field)
        runner.tap_frame = before
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        with patch.dict(sys.modules, {'module.campaign.structures': SimpleNamespace(facility_catalog=lambda r: catalog),
                                      'module.campaign.facility_memory': SimpleNamespace(
                                          FacilityUnconfirmed=type('FacilityUnconfirmed', (Exception,), {}))}):
            runner.remember_manual_entry((20, 0, 'EX', 1), self.target)
            field, position, stage = catalog.remember_manual.call_args.args
            self.assertIs(field, before)
            np.testing.assert_allclose(position, [600., 700.])
            self.ns['squad_arrow'].return_value = np.array([888., 400.])
            runner.remember_manual_entry((20, 0, 'EX', 2))
            field, position, stage = catalog.remember_manual.call_args.args
            self.assertIs(field, self.field)
            np.testing.assert_allclose(position, [888., 490.])
            self.ns['squad_arrow'].return_value = None

    def test_parse_only_complete_unique_stage(self):
        """任务必须含通关动词，多个编号和零值均不得作为关卡身份。"""
        parse = self.ns['parse_stage']
        for text in ('通关【20-6】', 'CLEAR [20－6]', '通關 20–6'):
            self.assertEqual(parse(text, quest=True), (20, 6))
        for text in ('20-6', '通关第20章', '通关20-6 / 20-7', '通关0-6', '通关20-0'):
            self.assertIsNone(parse(text, quest=True))

    def test_fixed_quest_line_bypasses_text_detector(self):
        """任务栏只直接识别裁好的单行，不再由文字检测器拆散动词和数字。"""
        reader = Mock()
        cv2.putText(self.field, 'CLEAR 20-6', (1570, 107), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (255, 255, 255), 1)
        reader.paddleocr.recognize.return_value = [dict(rec_text='通关【20-6】', rec_score=.94)]
        self.assertEqual(self.ns['read_stage'](self.field, reader, quest=True), (20, 6))
        reader.ocr.assert_not_called()
        for text, score in [('通关【20-6】', .89), ('20-6', .99), ('通关20-6', float('nan'))]:
            reader.paddleocr.recognize.return_value = [dict(rec_text=text, rec_score=score)]
            self.assertIsNone(self.ns['read_stage'](self.field, reader, quest=True))

    def test_portrait_quest_line_excludes_secondary_event_and_keeps_score_gate(self):
        """大厅使用独立单行区域；低分或无白字仍拒绝，不把活动任务混入主线。"""
        field = np.zeros((1280, 720, 3), np.uint8)
        cv2.putText(field, 'CLEAR 21-1', (452, 136), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 1)
        reader = Mock()
        reader.paddleocr.recognize.return_value = [dict(rec_text='通关【21-1]', rec_score=.914)]
        self.assertEqual(self.ns['read_quest_text'](field, reader), '通关【21-1]')
        reader.paddleocr.recognize.return_value = [dict(rec_text='通关【21-1]', rec_score=.899)]
        self.assertIsNone(self.ns['read_quest_text'](field, reader))

    def test_entry_title_requires_complete_anchor_and_preserves_branches(self):
        """Boss标签不当分支B，普通分支仍保留全号，EX/残缺/多编号均拒绝。"""
        parse = self.ns['parse_entry_title']
        for text, expected in [('20-31BOSS', (20, 31)), ('20-31 B0SS', (20, 31)),
                               ('20-23STAGE', (20, 23)), ('20-6B-1STAGE', (20, 6, 'B', 1)),
                               ('4-1 HARD STAGE', (4, 1)), ('4-9HARDBOSS', (4, 9))]:
            self.assertEqual(parse(text), expected)
        for text in ('20-31', '20-6BSTAGE', '20-6B-STAGE', 'EX STAGE', 'EX 20-6 STAGE',
                     '20-6STAGE20-7', '20-31BOSS extra', '0-31BOSS', '4-1 HARD'):
            self.assertIsNone(parse(text), text)

    def test_entry_title_fast_path_keeps_score_gate_and_legacy_fallback(self):
        """完整高分标题直接返回；低分和非有限分值不能绕过旧识别与开战身份保护。"""
        reader = Mock()
        reader.paddleocr.recognize.return_value = [dict(rec_text='20-31B0SS', rec_score=.97)]
        self.assertEqual(self.ns['read_stage'](self.field, reader), (20, 31))
        reader.ocr.assert_not_called()
        reader.ocr.return_value = dict(text='20-25STAGE')
        for score in (.89, float('nan')):
            reader.paddleocr.recognize.return_value = [dict(rec_text='20-31B0SS', rec_score=score)]
            self.assertEqual(self.ns['read_stage'](self.field, reader), (20, 25))
        self.assertEqual(reader.ocr.call_count, 2)

    def test_dedicated_title_keeps_observed_identity_and_confidence_gate(self):
        """新识别器使用实际完整标题；低分、坏编号和EX不能被期望任务修正或强行接受。"""
        reader = Mock()
        self.ns['read_title'].return_value = dict(rec_text='21-1STAGE', rec_score=.98)
        self.assertEqual(self.ns['read_stage'](self.field, reader), (21, 1))
        reader.paddleocr.recognize.assert_not_called()
        reader.ocr.assert_not_called()
        reader.paddleocr.recognize.return_value = []
        reader.ocr.return_value = dict(text='')
        for text, score in [('21-1STAGE', .899), ('21-1STAGE', float('nan')),
                            ('2H1STAGE', .99), ('EX STAGE', .99)]:
            self.ns['read_title'].return_value = dict(rec_text=text, rec_score=score)
            self.assertIsNone(self.ns['read_stage'](self.field, reader))

    def test_tight_title_fallback_preserves_anchor_and_score(self):
        """没有专用权重时，裁字回退仍保留完整标题与0.9门槛，不依靠当前任务补号。"""
        reader = Mock()
        self.ns['title_crop'].return_value = np.zeros((50, 200, 3), np.uint8)
        reader.paddleocr.recognize.side_effect = [[dict(rec_text='2H1STAGE', rec_score=.93)],
                                                [dict(rec_text='21-1STAGE', rec_score=.98)]]
        self.assertEqual(self.ns['read_stage'](self.field, reader), (21, 1))
        reader.ocr.assert_not_called()

    def test_entry_requires_two_matching_frames(self):
        """第一帧正确不放行；第二帧仍为预期关卡才允许战斗。"""
        self.ns['read_stage'] = Mock(return_value=(20, 6))
        guard = self.ns['EntryStageGuard'](self.session, (20, 6), self.owner.chapter_ocr)
        self.assertFalse(guard(self.field))
        self.assertTrue(guard(self.field))
        self.session.battle_step.assert_not_called()

    def test_wrong_entry_closes_without_battle(self):
        """连续读到错误关卡只关闭一次，并抛出不可降级成定位恢复的身份错误。"""
        self.ns['read_stage'] = Mock(return_value=(20, 7))
        guard = self.ns['EntryStageGuard'](self.session, (20, 6), self.owner.chapter_ocr)
        self.assertFalse(guard(self.field))
        with self.assertRaises(self.ns['StoryVerificationError']):
            guard(self.field)
        self.session.battle_step.assert_called_once_with(close_entry=True)
        self.executor.battle.assert_not_called()

    def test_unreadable_entry_times_out(self):
        """关卡号一直读不清时有独立十秒上限，不重置等待计时。"""
        self.ns['read_stage'] = Mock(return_value=None)
        guard = self.ns['EntryStageGuard'](self.session, (20, 6), self.owner.chapter_ocr)
        self.assertFalse(guard(self.field))
        self.ns['time'].monotonic.return_value = 11.
        with self.assertRaises(self.ns['StoryVerificationError']):
            guard(self.field)
        self.session.battle_step.assert_called_once_with(close_entry=True)

    def test_branch_identity_is_not_truncated(self):
        """分支保留完整编号；残缺分支和 EX 不能伪装成已知普通关卡。"""
        parse = self.ns['parse_stage']
        self.assertEqual(parse('据点防御战20-6B-1STAGE'), (20, 6, 'B', 1))
        for text in ('20-6B', '20-6B-', 'EX 20-6 STAGE', '20-6 / 20-7'):
            self.assertIsNone(parse(text), text)

    def test_visible_enemy_runs_battle_without_lobby_or_quest_advance(self):
        """实际 20-7 胜利可以继续任务 20-9；不要求每打一关任务栏都变化。"""
        self.ns['read_stage'] = Mock(return_value=(20, 9))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.inspect_entry = Mock(return_value=(20, 7))
        result = self.runner.run(ticks=1)
        self.assertEqual(result, dict(state='continue', reason='tick_budget'))
        self.session.tap.assert_called_once_with(self.target['click'])
        self.executor.battle.assert_called_once_with(dict(kind='story', id='story-20-7', stage=[20, 7]))
        self.runner.memory.victory.assert_called_once_with((20, 7))
        self.assertIsNone(self.session.entry_validator)
        self.session.snapshot.assert_not_called()
        self.reveal.assert_not_called()

    def test_branch_battle_is_supported_and_recorded_in_full(self):
        """当前章节的路边分支可清理，完整编号交给战斗及胜利观察簿。"""
        self.runner.quest = (20, 9)
        self.runner.inspect_entry = Mock(return_value=(20, 6, 'B', 1))
        self.assertTrue(self.runner.fight_entry(self.target))
        self.executor.battle.assert_called_once_with(dict(kind='story', id='story-20-6B-1', stage=[20, 6, 'B', 1]))
        self.runner.memory.identify.assert_called_once_with('ring-1', (20, 6, 'B', 1), 'encountered_on_route')
        self.runner.memory.victory.assert_called_once_with((20, 6, 'B', 1))

    def test_on_route_encounter_uses_actual_stage(self):
        """请求走向另一个怪但先碰到 20-7，只记录 20-7 胜利，不认领点击目标已清除。"""
        self.target['stage'] = (20, 9)
        self.runner.inspect_entry = Mock(return_value=(20, 7))
        self.runner.fight_entry(self.target)
        self.runner.memory.identify.assert_called_once_with('ring-1', (20, 7), 'encountered_on_route')
        self.runner.memory.victory.assert_called_once_with((20, 7))

    def test_cleared_stage_closes_without_start(self):
        """本次已确认胜利仍只关闭，避免重复挑战。"""
        self.owner.story_wins = {(20, 6)}
        self.runner.inspect_entry = Mock(return_value=(20, 6))
        self.runner.close_entry = Mock()
        self.assertFalse(self.runner.fight_entry(self.target))
        self.runner.close_entry.assert_called_once()
        self.executor.battle.assert_not_called()

    def test_locked_entry_is_deferred_without_start(self):
        """实录 23-27：前置主线未完成时进入战斗被锁，关闭并按 deferred 记下，不开战也不快速战斗。"""
        self.runner.quest = (20, 6)
        self.ns['goto'].entry_locked.return_value = True
        self.runner.inspect_entry = Mock(return_value=(20, 7))
        self.runner.close_entry = Mock()
        self.assertFalse(self.runner.fight_entry(self.target))
        self.runner.close_entry.assert_called_once()
        self.runner.memory.identify.assert_called_once_with('ring-1', (20, 7), 'deferred')
        self.executor.battle.assert_not_called()

    def test_guided_quest_behind_barrier_enters_switch_recovery_once(self):
        """实录24-27：任务已定位、没有可点敌人但两帧都有亮光幕，转入找机关流程而不是清怪巡检。"""
        self.runner.quest = (20, 6)
        self.runner.guided_quests.add((20, 6))
        self.runner.map_frame = Mock(return_value=self.field)
        barrier = dict(kind='barrier', state='curtain', confidence=.95, position=[900., 400.])
        objects = SimpleNamespace(detect_objects=Mock(return_value=[barrier]))
        structures = SimpleNamespace(recenter_squad=Mock(), StructureUnconfirmed=RuntimeError)
        with patch.dict(sys.modules, {'module.campaign.vision.objects': objects,
                                      'module.campaign.structures': structures}):
            self.assertTrue(self.runner.quest_barrier_blocked(self.field))
            self.assertEqual(self.runner.blocked_views, 1)
            structures.recenter_squad.assert_called_once()
            self.assertFalse(self.runner.quest_barrier_blocked(self.field))
            self.runner.quest = (20, 7)
            self.runner.guided_quests.add((20, 7))
            objects.detect_objects.return_value = [dict(barrier, state='pillar')]
            self.assertFalse(self.runner.quest_barrier_blocked(self.field))
            self.runner.guided_quests.discard((20, 7))
            objects.detect_objects.return_value = [barrier]
            self.assertFalse(self.runner.quest_barrier_blocked(self.field))
        self.session.tap.assert_not_called()

    def test_known_closed_switch_enters_recovery_before_sweep(self):
        """实录 2026-10-25（HARD 9）：没有可点敌人、目录里有没试过的关闭机关时先转找机关，同一道路状态只转一次。"""
        self.runner.quest = None
        catalog = SimpleNamespace(data=dict(state_epoch=0))
        pending = [('scene-0000', np.array([1300., 400.]))]
        structures = SimpleNamespace(facility_catalog=Mock(return_value=catalog), recenter_squad=Mock(),
                                     known_switches=Mock(return_value=pending))
        memory = SimpleNamespace(FacilityUnconfirmed=type('FacilityUnconfirmed', (RuntimeError,), {}))
        with patch.dict(sys.modules, {'module.campaign.structures': structures,
                                      'module.campaign.facility_memory': memory}):
            self.assertTrue(self.real_known_switch_blocked(self.field))
            self.assertEqual(self.runner.blocked_views, 1)
            self.assertFalse(self.runner.try_visible)
            structures.recenter_squad.assert_called_once()
            # 同一目标、同一道路状态不再转第二次，避免恢复没去试时空转。
            self.assertFalse(self.real_known_switch_blocked(self.field))
            # 开关踩亮后道路状态变化，可再转一次。
            catalog.data['state_epoch'] = 1
            self.assertTrue(self.real_known_switch_blocked(self.field))
            # 全部试过就照常巡检。
            structures.known_switches.return_value = []
            catalog.data['state_epoch'] = 2
            self.assertFalse(self.real_known_switch_blocked(self.field))
        self.assertEqual(structures.known_switches.call_args[0][1], 'quest:None')
        self.session.tap.assert_not_called()

    def test_hard_no_targets_hands_back_for_chapter_counter_once_per_win(self):
        """困难持续推图没怪：先交回读章节列表计数，不探索；读过后要有新胜利才再交回。"""
        self.runner.hard = True
        self.owner.progress_check = True
        self.owner.progress_checked_wins = None
        self.runner.scan = Mock(return_value=(self.field, []))
        self.runner.quest_target = Mock(return_value=(None, False))
        self.runner.reveal_marked_area = Mock(return_value=False)
        self.ns['chapter_identity'] = Mock(return_value=None)
        self.assertEqual(self.runner.run(ticks=1), dict(state='continue', reason='hard_progress_check'))
        self.runner.explore.assert_not_called()
        self.owner.progress_checked_wins = 0
        self.assertFalse(self.runner.progress_check_due())
        self.owner.story_wins.add((9, 27))
        self.assertTrue(self.runner.progress_check_due())
        self.owner.progress_check = False
        self.assertFalse(self.runner.progress_check_due())

    def test_no_targets_tries_known_switch_before_explore(self):
        """没有可点敌人时先问已知机关；转入找机关就不启动探索。"""
        self.runner.scan = Mock(return_value=(self.field, []))
        self.runner.quest_target = Mock(return_value=(None, False))
        self.runner.reveal_marked_area = Mock(return_value=False)
        self.runner.quest_barrier_blocked = Mock(return_value=False)
        self.runner.known_switch_blocked = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.runner.known_switch_blocked.assert_called()
        self.runner.explore.assert_not_called()

    def test_exhausted_explore_after_empty_quest_view_asks_for_manual_mechanism(self):
        """实录24-34：任务画面无敌人、探索看完也没有目标时，停止原因提示可能需要人工踩传送器等机关。"""
        self.ns['read_stage'] = Mock(return_value=(20, 18))
        self.runner.scan = Mock(return_value=(self.field, []))
        self.runner.quest_target = Mock(return_value=(None, False))
        self.runner.reveal_marked_area = Mock(return_value=False)
        self.runner.quest_barrier_blocked = Mock(return_value=False)
        with self.assertRaises(self.ns['StoryVerificationError']) as caught:
            self.runner.run(ticks=1)
        self.assertEqual(caught.exception.reason, 'story_exploration_exhausted')
        self.assertIn('探索 3 屏', str(caught.exception))
        self.assertIn('20-18', str(caught.exception))
        self.assertIn('人工查看', str(caught.exception))
        self.session.tap.assert_not_called()

    def test_explore_switch_only_steps_on_it(self):
        """实录 HARD 9：没有敌人、探索只找到关闭机关时去踩它，不再停在只找敌人的巡检。"""
        self.runner.scan = Mock(return_value=(self.field, []))
        self.runner.quest_target = Mock(return_value=(None, False))
        self.runner.reveal_marked_area = Mock(return_value=False)
        self.runner.quest_barrier_blocked = Mock(return_value=False)
        found = dict(state='switch', target=dict(kind='ground_mechanism', state='off', position=[900., 600.]),
                     field=self.field)
        self.runner.explore = Mock(return_value=found)
        self.runner.step_on_switch = Mock(return_value='attempted')
        self.assertEqual(self.runner.run(ticks=1)['reason'], 'tick_budget')
        self.runner.step_on_switch.assert_called_once_with(found)

    def test_step_on_switch_respects_interaction_budget(self):
        """每次真实进展前设施交互最多两次；用完后探索只找到机关就停止，不反复踩。"""
        structures = SimpleNamespace(STRUCTURE_INPUTS=2, attempt_structure=Mock(return_value='attempted'),
                                     facility_catalog=Mock(return_value=None))
        found = dict(state='switch', target=dict(kind='ground_mechanism'), field=self.field)
        with patch.dict(sys.modules, {'module.campaign.structures': structures}):
            self.assertEqual(self.runner.step_on_switch(found), 'attempted')
            for _ in range(2):
                self.runner.world.bump('structure_input', reset_on=('win', 'switch'))
            with self.assertRaises(self.ns['StoryVerificationError']):
                self.runner.step_on_switch(found)
        structures.attempt_structure.assert_called_once()

    def test_scene_marker_without_ring_walks_to_it_once(self):
        """实录24-37：菱形下的 Boss 入口没认出来时点菱形交给原生寻路，到备战再核对；同一任务只走一次。"""
        self.runner.quest = (20, 6)
        marker = dict(kind='scene', position=[634., 266.], box=[608, 242, 658, 290], confidence=.93)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.assertTrue(self.runner.walk_to_marker(self.field))
        self.session.tap.assert_called_once_with([634., 266.])
        self.runner.fight_entry.assert_called_once_with()
        self.assertFalse(self.runner.walk_to_marker(self.field))
        self.session.tap.reset_mock()
        # 离屏/提示形态、位置跳开都不点。
        self.runner.quest = (20, 7)
        self.ns['story_markers'].return_value = [dict(marker, kind='edge')]
        self.assertFalse(self.runner.walk_to_marker(self.field))
        self.ns['story_markers'].side_effect = [[marker], [dict(marker, position=[900., 266.])]] + [[]]*4
        self.assertFalse(self.runner.walk_to_marker(self.field))
        # 动画让中间几帧漏检时，最多看6帧，两帧同位置即确认。
        self.ns['story_markers'].side_effect = [[marker], [], [], [dict(marker, position=[636., 268.])]]
        self.assertTrue(self.runner.walk_to_marker(self.field))
        self.session.tap.assert_called_once_with([636., 268.])
        self.session.tap.reset_mock()
        self.session.tap.assert_not_called()

    def test_destination_quest_on_map_returns_to_lobby_without_clicking(self):
        """实录23章通关：地图任务为「前往【方舟】的【市内】」时连续两帧确认后交回大厅，不在地图点击。"""
        texts = iter(['前往【方舟】的【市内】', '前往【方舟】的【市内】'])
        self.ns['read_quest_text'] = Mock(side_effect=lambda field, reader, minimum=.9: next(texts))
        self.assertTrue(self.runner.destination_return(self.field))
        self.assertEqual(self.ns['read_quest_text'].call_args.kwargs['minimum'], self.ns['DESTINATION_SCORE'])
        self.session.tap.assert_not_called()
        for pair in (['前往【方舟】的【市内】', '前往【方舟】的【简报室】'], ['通关【24-1】', '通关【24-1】']):
            texts = iter(pair)
            self.assertFalse(self.runner.destination_return(self.field))

    def test_all_current_chapter_enemies_are_allowed(self):
        """清怪范围独立于任务号：当前章较后编号只要实际可进入，也在清理范围。"""
        self.runner.quest = (20, 9)
        self.runner.inspect_entry = Mock(return_value=(20, 10))
        self.assertTrue(self.runner.fight_entry(self.target))
        self.executor.battle.assert_called_once()

    def test_ex_is_kept_pending_while_ordinary_candidates_remain_selectable(self):
        """EX 不支持手动战斗时不抢占普通目标，也不把它算作已清或没有敌人。"""
        self.assertFalse(self.runner.selectable(dict(self.target, kind='ex')))
        self.assertTrue(self.runner.manual_ex_seen)
        self.assertTrue(self.runner.selectable(self.target))
        self.runner.inspect_entry = Mock(return_value=(20, 0, 'EX', 1))
        self.runner.close_entry = Mock()
        self.assertFalse(self.runner.fight_entry(self.target))
        self.runner.close_entry.assert_called_once()
        self.runner.memory.identify.assert_called_once_with('ring-1', (20, 0, 'EX', 1), 'manual_required')
        self.runner.remember_manual_entry.assert_called_once_with((20, 0, 'EX', 1), self.target)
        self.executor.battle.assert_not_called()
        self.runner.memory.victory.assert_not_called()

    def test_other_chapter_stops_without_start(self):
        """备战显示另一章时关闭并停止，不将候选顺序当作开战许可。"""
        self.runner.inspect_entry = Mock(return_value=(21, 1))
        self.runner.close_entry = Mock()
        with self.assertRaises(self.ns['StoryVerificationError']):
            self.runner.fight_entry(self.target)
        self.runner.close_entry.assert_called_once()
        self.executor.battle.assert_not_called()

    def test_empty_view_explores_then_targets_found_enemy(self):
        """视野没有怪时探索；探索找到的敌人用探索那一帧作为点击依据，零可见敌人不作为换章理由。"""
        self.ns['read_stage'] = Mock(return_value=(20, 6))
        self.runner.scan = Mock(return_value=(self.field, []))
        self.runner.explore = Mock(return_value=dict(state='enemy', targets=[self.target], field=self.field))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.inspect_entry = Mock(return_value=(20, 7))
        self.assertEqual(self.runner.run(ticks=1)['reason'], 'tick_budget')
        self.session.tap.assert_called_once_with([600., 700.])

    def test_quest_locator_requires_blue_control_and_right_arrow(self):
        """右箭头提供真实控件坐标，左箭头或缺少蓝底均不能触发任务定位。"""
        shape = np.zeros((32, 32), np.uint8)
        cv2.fillPoly(shape, [np.array([[0, 8], [15, 8], [15, 0], [31, 15],
                                     [15, 31], [15, 23], [0, 23]])], 255)
        glyph = cv2.resize(shape, (14, 12), interpolation=cv2.INTER_NEAREST)
        for blue, right in ((True, True), (True, False), (False, True)):
            field = self.field.copy()
            if blue:
                field[85:113, 1734:1765] = (220, 160, 30)
            patch = field[94:106, 1742:1756]
            patch[(glyph if right else np.fliplr(glyph)) > 0] = 255
            point = self.ns['quest_jump_point'](field)
            if blue and right:
                self.assertIsNotNone(point)
                np.testing.assert_allclose(point, [1749., 100.], atol=1.)
            else:
                self.assertIsNone(point)

    def test_return_lobby_requires_exact_text_and_final_frame_control(self):
        """非数字任务独立解析，控件改变后的最终帧不能沿用旧返回点击。"""
        parse = self.ns['return_lobby_quest']
        for text in ('返回【大厅】', '返回[大廳]', '返回大厅'):
            self.assertTrue(parse(text))
        for text in (None, '返回【大厅', '请返回大厅领取', '通关20-31', '前往大厅'):
            self.assertFalse(parse(text))
        self.ns['read_quest_text'] = Mock(return_value='返回【大厅】')
        self.ns['quest_jump_point'] = Mock(return_value=(1749., 100.))
        self.assertTrue(self.runner.return_lobby(self.field))
        point, guard = self.session.tap_verified_ui.call_args.args
        self.assertEqual(point, (1749., 100.))
        self.ns['read_quest_text'].return_value = '通关【21-1】'
        self.assertFalse(guard(self.field))
        self.executor.battle.assert_not_called()

    def test_return_lobby_precedes_scene_scanning_and_keeps_unstable_quest_blocked(self):
        """返回任务优先交大厅处理；两帧变化时既不点控件，也不继续在旧图扫怪。"""
        self.ns['read_stage'] = Mock(return_value=None)
        self.ns['read_quest_text'] = Mock(return_value='返回【大厅】')
        self.ns['quest_jump_point'] = Mock(return_value=(1749., 100.))
        self.runner.scan = Mock()
        self.assertEqual(self.runner.run(ticks=1), dict(state='continue', reason='story_return_lobby', chapter=20))
        self.runner.scan.assert_not_called()
        self.session.tap_verified_ui.reset_mock()
        self.ns['read_quest_text'].side_effect = ['返回【大厅】', '通关【21-1】']
        with self.assertRaises(self.ns['StoryVerificationError']):
            self.runner.return_lobby(self.field)
        self.session.tap_verified_ui.assert_not_called()

    def test_quest_guide_never_clicks_in_level(self):
        """实录 2026-10-23：任务蓝箭头局内不移动镜头；只登记已核对，不发送输入，扫描照常进行。"""
        self.runner.quest = (20, 18)
        self.ns['quest_jump_point'] = Mock(return_value=(1749., 100.))
        self.ns['read_stage'] = Mock(return_value=(20, 18))
        self.assertFalse(self.runner.guide_quest(self.field))
        self.assertIn((20, 18), self.runner.guided_quests)
        self.session.tap_verified_ui.assert_not_called()
        self.session.tap.assert_not_called()

    def test_stable_main_marker_prioritizes_existing_ring_and_keeps_hidden_target_as_hint(self):
        """任务只关联已有脚圈；脚圈被底栏排除或标记漂移时，不能用图标虚构点击。"""
        self.runner.quest = (20, 18)
        marker = dict(kind='scene', position=[1000., 650.], box=[965, 615, 1035, 685], confidence=.95)
        self.ns['story_markers'] = Mock(return_value=[marker])
        near = dict(self.target, id='near-but-blocked', click=[1200., 300.])
        marked = dict(self.target, id='marked-ring', click=[1000., 750.])
        target, hinted = self.runner.quest_target(self.field, self.field, [near, marked])
        self.assertEqual(target['id'], 'marked-ring')
        self.assertTrue(hinted)
        target, hinted = self.runner.quest_target(self.field, self.field, [near])
        self.assertIsNone(target)
        self.assertTrue(hinted)
        self.ns['story_markers'].side_effect = [[marker], [dict(marker, position=[1040., 650.])]]
        self.assertEqual(self.runner.quest_target(self.field, self.field, [near, marked]), (None, False))

    def test_visible_marked_ring_can_precede_unrelated_facility_recovery(self):
        """旧目标受阻后，独立确认的新任务脚圈优先于平台试点，不绕过实际备战核验。"""
        self.runner.quest = (20, 18)
        self.runner.guided_quests.add(self.runner.quest)
        self.runner.blocked_views = 1
        self.ns['read_stage'] = Mock(return_value=self.runner.quest)
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.quest_target = Mock(return_value=(self.target, True))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.session.tap.assert_called_once_with(self.target['click'])
        self.runner.fight_entry.assert_called_once_with(self.target)

    def test_low_scene_marker_only_moves_camera_and_requires_new_scan(self):
        """脚圈被底栏遮住时用稳定菱形规划镜头，不点击菱形或启动未核对战斗。"""
        marker = dict(kind='scene', position=[900., 830.], confidence=.95)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.ns['squad_arrow'].return_value = [1100., 400.]
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        args, kwargs = self.ns['pan_scene'].call_args
        np.testing.assert_allclose(args[1], [-12., -380.])
        np.testing.assert_allclose(kwargs['squad_ground'], [1100., 490.])
        self.session.tap.assert_not_called()
        self.executor.battle.assert_not_called()
        for _ in range(3):
            self.runner.world.bump('marker_pan')
        self.assertFalse(self.runner.reveal_marked_area(self.field))
        self.runner.world.advance('win')
        self.assertTrue(self.runner.reveal_marked_area(self.field))

    def test_top_bar_marker_only_pans_camera(self):
        """实录 25-18：菱形贴在顶栏右侧，只按它的方向平移镜头，不在顶栏点击。"""
        marker = dict(kind='hud', position=[1617., 53.], confidence=.91)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.ns['squad_arrow'].return_value = [888., 410.]
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.assertEqual(self.ns['story_markers'].call_args.kwargs, dict(top_bar=True))
        args, kwargs = self.ns['pan_scene'].call_args
        drag = np.asarray(args[1])
        self.assertAlmostEqual(float(np.linalg.norm(drag)), 480., places=3)
        self.assertTrue(drag[0] < 0 < drag[1])
        self.session.tap_verified_ui.assert_not_called()
        self.session.tap.assert_not_called()

    def test_far_edge_marker_is_tapped_once_to_move_camera_to_target(self):
        """实录21-15：贴在顶栏下的离屏菱形点一次由游戏移镜头，最多两次后回到小步平移。"""
        marker = dict(kind='edge', position=[651., 225.], confidence=.95)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.ns['squad_arrow'].return_value = [1100., 400.]
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.session.tap_verified_ui.assert_called_once()
        point, validate = self.session.tap_verified_ui.call_args.args
        self.assertEqual(point, [651., 225.])
        self.assertTrue(validate(self.field))
        self.ns['pan_scene'].assert_not_called()
        self.session.tap.assert_not_called()
        self.runner.world.bump('marker_jump', reset_on=('win',))
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.ns['pan_scene'].assert_called_once()
        self.assertEqual(self.session.tap_verified_ui.call_count, 1)

    def test_in_scene_marker_near_top_is_panned_not_tapped(self):
        """实录21-32：y≈159的场内菱形不是顶栏夹住的边缘菱形，点了会触发原生寻路，只能平移镜头。"""
        marker = dict(kind='scene', position=[499., 159.], confidence=.94)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.ns['squad_arrow'].return_value = [1100., 400.]
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.session.tap_verified_ui.assert_not_called()
        self.ns['pan_scene'].assert_called_once()

    def test_blinking_edge_marker_failing_final_check_is_not_a_run_stop(self):
        """实录右下边缘菱形闪烁：输入前复核失败只放弃这次点击，不抛出停止整轮的窗口异常。"""
        marker = dict(kind='edge', position=[1688., 828.], confidence=.91)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.ns['squad_arrow'].return_value = [1100., 400.]
        self.session.tap_verified_ui.side_effect = self.ns['UiControlChanged']('blink')
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.assertEqual(self.runner.world.count('marker_jump', reset_on=('win',)), 1)
        self.ns['pan_scene'].assert_not_called()

    def test_unanchored_marker_with_visible_enemies_returns_to_main_loop(self):
        """实录21-32：小队与普通敌人叠在一起、头顶标记缺失；有可点击脚圈时不进入无锚点设施恢复。"""
        marker = dict(kind='scene', position=[669., 154.], confidence=.96)
        self.ns['story_markers'] = Mock(return_value=[marker])
        self.ns['squad_arrow'].return_value = None
        self.ns['squad_offscreen'] = Mock(return_value=None)
        self.assertFalse(self.runner.reveal_marked_area(self.field, has_targets=True))
        self.ns['pan_scene'].assert_not_called()
        self.session.tap.assert_not_called()

    def test_changing_marker_does_not_move_camera(self):
        """两帧标记不一致时不按旧菱形规划镜头。"""
        self.ns['story_markers'] = Mock(side_effect=[
            [dict(kind='scene', position=[900., 830.], confidence=.95)], [], [], [], [], []])
        self.assertFalse(self.runner.reveal_marked_area(self.field))
        self.ns['pan_scene'].assert_not_called()

    def test_blinking_marker_requires_two_fresh_detections_within_bounded_probe(self):
        """发光导致初帧和中间帧漏检时仍须两次同位置识别，单次闪现不发送镜头手势。"""
        marker = dict(kind='cue', position=[475., 828.], confidence=.95)
        self.ns['squad_arrow'].return_value = [1100., 400.]
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        self.ns['story_markers'] = Mock(side_effect=[[], [marker], [], [marker]])
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.assertEqual(self.runner.map_frame.call_count, 3)
        self.ns['pan_scene'].assert_called_once()
        self.session.tap.assert_not_called()
        self.runner.world.advance('win')
        self.runner.map_frame.reset_mock()
        self.ns['pan_scene'].reset_mock()
        self.ns['story_markers'] = Mock(side_effect=[[], [marker], [], [], [], []])
        self.assertFalse(self.runner.reveal_marked_area(self.field))
        self.assertEqual(self.runner.map_frame.call_count, 5)
        self.ns['pan_scene'].assert_not_called()

    def test_marker_kind_animation_does_not_turn_cue_into_ground_click(self):
        """两帧仅场内／离屏分类波动但位置稳定时仍可安全调整镜头，不点击可能是场景物的标记。"""
        self.ns['story_markers'] = Mock(side_effect=[
            [dict(kind='scene', position=[900., 830.], confidence=.95)],
            [dict(kind='edge', position=[901., 830.], confidence=.96)]])
        self.ns['squad_arrow'].return_value = [1100., 400.]
        self.ns['goto'].ARROW_GROUND_OFFSET = 90.
        self.assertTrue(self.runner.reveal_marked_area(self.field))
        self.reveal.assert_not_called()
        self.session.tap.assert_not_called()
        self.ns['pan_scene'].assert_called_once()

    def test_automatic_transition_returns_to_uncleared_chapter(self):
        """指定单章清理时游戏自动前进仍返回旧章，不将换章当清空证明。"""
        self.owner.continuous = False
        self.ns['chapter_identity'].return_value = (21, 'normal')
        self.runner.restore_chapter = Mock()
        self.runner.run(ticks=1)
        self.runner.restore_chapter.assert_called_once_with((21, 'normal'))
        self.runner.memory.observe.assert_not_called()

    def test_continuous_automatic_transition_keeps_pending_and_accepts_two_frames(self):
        """连续主线确认两帧新章后继续，不返回旧章，也不将其标成清空。"""
        self.ns['chapter_identity'].return_value = (21, 'normal')
        self.runner.restore_chapter = Mock()
        result = self.runner.run(ticks=1)
        self.assertEqual(result['reason'], 'chapter_changed')
        self.runner.restore_chapter.assert_not_called()
        self.runner.journal.complete.assert_not_called()
        self.session.tap.assert_not_called()

    def test_mainline_next_chapter_needs_fresh_quest_and_preserves_clearance(self):
        """只有重复确认的新章主线可触发一次切章；最终输入校验及两帧新章身份仍有效。"""
        self.ns['read_stage'] = Mock(return_value=(21, 1))
        self.ns['chapter_identity'].side_effect = [(20, 'normal')] * 3 + [(21, 'normal')] * 2
        def tap(point, validator):
            """模拟输入闸门在最后一帧再次执行生产校验。"""
            self.assertEqual(point, (1715, 940))
            self.assertTrue(validator(self.field))
        self.session.tap_verified_ui.side_effect = tap
        result = self.runner.follow_mainline(self.field, (21, 1))
        self.assertEqual(result['reason'], 'chapter_changed')
        self.session.tap_verified_ui.assert_called_once()
        self.runner.journal.complete.assert_not_called()
        self.session.tap_verified_ui.reset_mock()
        self.ns['chapter_identity'].side_effect = None
        self.ns['chapter_identity'].return_value = (20, 'normal')
        self.ns['read_stage'].return_value = (20, 31)
        self.assertIsNone(self.runner.follow_mainline(self.field, (21, 1)))
        self.session.tap_verified_ui.assert_not_called()

    def test_mainline_recheck_tolerates_one_flaky_quest_frame(self):
        """实录22→23：复核帧任务 OCR 偶尔读空；重读下一帧通过即切章。"""
        self.ns['read_stage'] = Mock(side_effect=[None, (21, 1), (21, 1)])
        self.ns['chapter_identity'].side_effect = [(20, 'normal')] * 5 + [(21, 'normal')] * 2
        self.session.tap_verified_ui.side_effect = lambda point, validator: validator(self.field)
        self.assertEqual(self.runner.follow_mainline(self.field, (21, 1))['reason'], 'chapter_changed')

    def test_next_chapter_quest_never_sweeps_old_chapter(self):
        """持续模式任务已在下一章但切章核对不过：不扫描旧章，三轮后停止并说明原因。"""
        self.ns['read_stage'] = Mock(return_value=(21, 1))
        self.runner.follow_mainline = Mock(return_value=None)
        self.runner.scan = Mock()
        with self.assertRaises(self.ns['StoryVerificationError']):
            self.runner.run(ticks=5)
        self.assertEqual(self.runner.follow_mainline.call_count, 3)
        self.runner.scan.assert_not_called()

    def test_ex_identity_requires_title_and_is_not_silently_skipped(self):
        """EX 连续两帧标题通过后进入清怪范围；使用内部序号且不虚构普通关卡号。"""
        self.ns['goto'].battle_popup_score.return_value = 1.
        self.ns['battle'].ex_stage.return_value = True
        self.ns['read_ex_entry'] = Mock(return_value=True)
        stage = self.runner.inspect_entry()
        self.assertEqual(stage, (20, 0, 'EX', 1))
        self.assertEqual(self.ns['read_ex_entry'].call_count, 2)
        self.assertEqual(self.ns['stage_text'](stage), '20-EX#1')
        self.session.battle_step.assert_not_called()

    def test_movement_timeout_does_not_repeat_click(self):
        """可靠停滞请求诊断，单轮只发送一个落点且不宣称道路不可达。"""
        self.ns['read_stage'] = Mock(return_value=(20, 6))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.wait_entry = Mock(return_value=False)
        self.assertEqual(self.runner.run(ticks=1)['state'], 'continue')
        self.session.tap.assert_called_once()
        self.runner.memory.attempt.assert_called_once_with('ring-1', 'stalled', None)
        self.runner.memory.identify.assert_not_called()
        self.executor.battle.assert_not_called()

    def test_click_feedback_keeps_waiting_for_observed_motion_past_twenty_seconds(self):
        """真实移动可超过20秒，途中备战仍优先接管，不追加第二次落点。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='moving', reason='camera_compensated')
        self.ns['goto'].battle_popup_score.side_effect = lambda frame: 1. if clock[0] >= 25. else 0.
        self.assertTrue(self.runner.wait_entry(self.field))
        self.assertGreaterEqual(clock[0], 25.)
        self.session.tap.assert_not_called()

    def test_unobserved_click_is_treated_as_stall_without_facility_input(self):
        """一直缺失位移观测：停止策略分级后按停滞交回主循环（换目标或受阻流程），不发送追加点击。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='unknown', reason='squad_unobserved')
        self.assertFalse(self.runner.wait_entry(self.field))
        self.assertEqual(self.runner.entry_outcome['outcome'], 'observation_unknown')
        self.runner.memory.identify.assert_not_called()
        self.session.tap.assert_not_called()

    def test_unobserved_click_recenters_on_offscreen_marker_once(self):
        """实录23-11：小队走出画面但离屏头像可见，找回镜头一次后继续观察，备战出现即接管。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='unknown', reason='squad_unobserved')
        self.ns['squad_offscreen'].return_value = dict(center=[996., 223.], direction=270.)
        recentered = []
        recenter = Mock(side_effect=lambda runner: recentered.append(clock[0]))
        self.ns['goto'].battle_popup_score.side_effect = lambda frame: 1. if recentered and clock[0] > recentered[0]+3 else 0.
        structures = SimpleNamespace(StructureUnconfirmed=RuntimeError, recenter_squad=recenter)
        with patch.dict(sys.modules, {'module.campaign.structures': structures}):
            self.assertTrue(self.runner.wait_entry(self.field))
        recenter.assert_called_once_with(self.runner)
        self.session.tap.assert_not_called()

    def test_unobserved_after_recenter_is_stall(self):
        """找回镜头后仍看不到小队：只找回一次，按停滞交回主循环。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='unknown', reason='squad_unobserved')
        self.ns['squad_offscreen'].return_value = dict(center=[996., 223.], direction=270.)
        recenter = Mock()
        structures = SimpleNamespace(StructureUnconfirmed=RuntimeError, recenter_squad=recenter)
        with patch.dict(sys.modules, {'module.campaign.structures': structures}):
            self.assertFalse(self.runner.wait_entry(self.field))
        recenter.assert_called_once_with(self.runner)

    def test_unobserved_static_click_is_no_effect(self):
        """实录 2026-10-24（25章）：点了自己脚下的橙色圈，二十秒小队和小地图都看不到、镜头没动：当作点了没反应。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='unknown', reason='squad_unobserved')
        self.ns['static_view'].return_value = True
        self.assertFalse(self.runner.wait_entry(self.field, target=self.target))
        self.assertTrue(self.runner.entry_outcome['no_effect'])
        self.assertEqual(self.runner.entry_outcome['outcome'], 'observation_unknown')
        self.session.tap.assert_not_called()

    def test_unobserved_moving_view_is_stall_not_no_effect(self):
        """镜头动过（或配准不上）就不能说点了没反应：按停滞处理（会进入受阻流程），不标 no_effect。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='unknown', reason='squad_unobserved')
        self.assertFalse(self.runner.wait_entry(self.field, target=self.target))
        self.assertNotIn('no_effect', self.runner.entry_outcome)

    def test_no_effect_click_is_rejected_without_facility_recovery(self):
        """点了没反应：记下位置、本目标暂缓，但不算道路受阻，不进入设施诊断。"""
        self.ns['read_stage'] = Mock(return_value=(25, 18))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))

        def no_effect(*args, **kwargs):
            self.runner.entry_outcome = dict(state='unknown', outcome='observation_unknown', no_effect=True)
            return False
        self.runner.wait_entry = Mock(side_effect=no_effect)
        self.runner.run(ticks=1)
        self.assertEqual(len(self.runner.rejected), 1)
        self.assertEqual(self.runner.rejected.records[0]['reason'], 'no_effect')
        self.assertEqual((self.runner.blocked_views, self.runner.try_visible), (0, False))
        self.assertIn(self.target['id'], self.runner.deferred)
        self.assertEqual(self.runner.memory.attempt.call_args[0][:2], ('ring-1', 'no_effect'))

    def test_stall_records_position_and_blocked_win_clears_it(self):
        """停滞也记下位置；受阻期间打赢可见敌人后道路可能已通，记录清空允许重试。"""
        self.ns['read_stage'] = Mock(return_value=(25, 15))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.wait_entry = Mock(return_value=False)
        self.runner.run(ticks=1)
        self.assertEqual(self.runner.rejected.records[0]['reason'], 'stalled')
        other = dict(self.target, id='ring-9', click=[1200., 600.])
        self.runner.scan = Mock(return_value=(self.field, [other]))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.assertEqual(len(self.runner.rejected), 0)

    def test_stall_after_moving_retries_once_then_rejects(self):
        """实录 2026-10-25（HARD 9）：点 Boss 后经电梯被送上山脊停下；走过才停的目标先不记走不到、不进受阻，
        下一轮从新位置再点一次，同一位置第二次停下才记走不到。"""
        self.ns['read_stage'] = Mock(return_value=(25, 15))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        # 场景配准换成恒等投影，只测记录规则。
        self.runner.rejected.points = lambda field: [(record['point'], record['reason'])
                                                     for record in self.runner.rejected.records]

        def moved_then_stopped(*args, **kwargs):
            self.runner.entry_outcome = dict(state='stationary', outcome='stalled', moved=True)
            return False
        self.runner.wait_entry = Mock(side_effect=moved_then_stopped)
        self.runner.run(ticks=1)
        self.assertEqual([record['reason'] for record in self.runner.rejected.records], ['moved'])
        self.assertEqual(self.runner.blocked_views, 0)
        self.assertNotIn(self.target['id'], self.runner.deferred)
        self.assertEqual(self.runner.memory.attempt.call_args[0][:2], ('ring-1', 'moved'))
        self.runner.run(ticks=1)
        self.assertEqual(self.session.tap.call_count, 2)
        self.assertEqual([record['reason'] for record in self.runner.rejected.records], ['moved', 'stalled'])
        self.assertEqual(self.runner.blocked_views, 1)
        self.assertIn(self.target['id'], self.runner.deferred)

    def test_rejected_position_skips_ring_with_new_id(self):
        """同一处换了扫描编号仍跳过，改点画面内其他目标，并留下跳过证据。"""
        self.ns['read_stage'] = Mock(return_value=(25, 15))
        renamed = dict(self.target, id='ring-40-0')
        other = dict(self.target, id='ring-40-1', click=[1200., 600.])
        self.runner.scan = Mock(return_value=(self.field, [renamed, other]))
        self.runner.rejected = Mock(filter=Mock(return_value=([other], [(renamed, 8., 'no_effect')])))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.session.tap.assert_called_once_with(other['click'])
        skipped = [call for call in self.session.record_evidence.call_args_list if call[0][0] == 'rejected_skip']
        self.assertEqual(skipped[0][1]['skipped'][0]['id'], 'ring-40-0')

    def test_structure_scout_keeps_recovery_priority_over_another_enemy_click(self):
        """受阻后的设施换视角仍需继续找设施，不立即插入另一次敌人点击打断恢复。"""
        self.runner.blocked_views = 1
        self.ns['read_stage'] = Mock(return_value=(20, 25))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        recover = Mock(return_value='scouted')
        with patch.dict(sys.modules, {'module.campaign.structures':
                                     SimpleNamespace(recover_visible_structure=recover)}):
            self.runner.run(ticks=1)
        self.assertEqual(self.runner.blocked_views, 1)
        recover.assert_called_once_with(self.runner, self.field)
        self.session.tap.assert_not_called()
        self.executor.battle.assert_not_called()

    def test_fresh_stall_tries_other_visible_enemy_before_facility_recovery(self):
        """实录22-27：刚受阻时画面内还有可选敌人（挡在机关前的支线怪），先打它；打赢后重试原目标。"""
        self.runner.blocked_views, self.runner.try_visible = 1, True
        self.runner.deferred.add('blocked-goal')
        self.ns['read_stage'] = Mock(return_value=(20, 25))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        recover = Mock(return_value='scouted')
        with patch.dict(sys.modules, {'module.campaign.structures':
                                     SimpleNamespace(recover_visible_structure=recover)}):
            self.runner.run(ticks=1)
        recover.assert_not_called()
        self.session.tap.assert_called_once_with(self.target['click'])
        self.assertEqual((self.runner.blocked_views, self.runner.try_visible), (0, False))
        self.assertNotIn('blocked-goal', self.runner.deferred)

    def test_stall_marks_visible_alternatives_then_recovery_keeps_priority(self):
        """停滞后标记先试可见敌人；没有其他可选目标时进入恢复，恢复开始后不再插入敌人点击。"""
        self.ns['read_stage'] = Mock(return_value=(20, 25))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.wait_entry = Mock(return_value=False)
        self.runner.run(ticks=1)
        self.assertEqual((self.runner.blocked_views, self.runner.try_visible), (1, True))
        self.assertIn(self.target['id'], self.runner.deferred)
        recover = Mock(return_value='scouted')
        with patch.dict(sys.modules, {'module.campaign.structures':
                                     SimpleNamespace(recover_visible_structure=recover)}):
            self.runner.run(ticks=1)
        recover.assert_called_once()
        self.assertFalse(self.runner.try_visible)

    def test_blocked_prefers_enemy_guarding_closed_switch(self):
        """实录 2026-10-24（25章）：受阻后先打站在关闭机关上的怪，而不是离小队最近、走不到的那只。"""
        self.runner.blocked_views, self.runner.try_visible = 1, True
        self.ns['read_stage'] = Mock(return_value=(25, 8))
        guard = dict(self.target, id='ring-27-2', click=[1040., 641.])
        near = dict(self.target, id='ring-31-0', click=[900., 520.])
        self.runner.scan = Mock(return_value=(self.field, [near, guard]))
        self.runner.closed_switch_points = Mock(return_value=[np.array([1022.7, 626.1])])
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.session.tap.assert_called_once_with(guard['click'])

    def test_switch_priority_only_while_blocked(self):
        """没受阻时照常按离小队远近选目标，不为排序去配准设施目录。"""
        self.ns['read_stage'] = Mock(return_value=(25, 8))
        guard = dict(self.target, id='ring-27-2', click=[1040., 641.])
        near = dict(self.target, id='ring-31-0', click=[900., 520.])
        self.runner.scan = Mock(return_value=(self.field, [guard, near]))
        self.runner.closed_switch_points = Mock(return_value=[np.array([1022.7, 626.1])])
        self.runner.wait_entry = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.runner.closed_switch_points.assert_not_called()
        self.session.tap.assert_called_once_with(near['click'])

    def test_closed_switch_points_projects_catalog(self):
        """目录里仍关闭的机关按本帧投影返回；目录配准失败时返回空，照常按远近选目标。"""
        catalog = SimpleNamespace(observe=Mock(), expected=[dict(position=[1022.7, 626.1])])
        structures = SimpleNamespace(facility_catalog=Mock(return_value=catalog))
        unconfirmed = type('FacilityUnconfirmed', (RuntimeError,), {})
        memory = SimpleNamespace(FacilityUnconfirmed=unconfirmed)
        with patch.dict(sys.modules, {'module.campaign.structures': structures,
                                      'module.campaign.facility_memory': memory}):
            points = self.real_closed_switch_points(self.field)
            self.assertEqual([point.tolist() for point in points], [[1022.7, 626.1]])
            catalog.observe.side_effect = unconfirmed('no anchor')
            self.assertEqual(self.real_closed_switch_points(self.field), [])

    def test_stall_followup_popup_still_fights(self):
        """判停滞后补看到备战弹窗：照常核对开战，不把目标记成走不到。"""
        self.ns['read_stage'] = Mock(return_value=(25, 8))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.runner.wait_entry = Mock(return_value=False)
        self.runner.stall_followup = Mock(return_value=True)
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.runner.fight_entry.assert_called_once_with(self.target)
        self.assertNotIn(self.target['id'], self.runner.deferred)
        self.assertEqual(self.runner.blocked_views, 0)

    def test_stall_followup_records_frame(self):
        """补看帧总要留证；只有备战弹窗才算走到。"""
        self.runner.entry_outcome = dict(state='stationary')
        self.assertFalse(self.runner.stall_followup(self.target))
        event = self.session.record_evidence.call_args
        self.assertEqual(event.args[0], 'native_stall_followup')
        self.assertFalse(event.kwargs['popup'])
        self.ns['goto'].battle_popup_score.return_value = 1.
        self.assertTrue(self.runner.stall_followup(self.target))
        self.assertEqual(self.runner.entry_outcome, dict(state='entry'))

    def test_click_feedback_frames_survive_slot_wraparound(self):
        """寻路反馈与停滞补看另存不回绕副本，32 槽覆盖后仍能回看。"""
        with tempfile.TemporaryDirectory() as folder:
            evidence = ActionEvidence(Path(folder), capacity=2)
            evidence.record('native_click_feedback', self.field)
            evidence.record('native_stall_followup', self.field)
            for _ in range(4):
                evidence.record('mouse_click', self.field)
            kept = sorted(path.name for path in (evidence.folder/'keep').glob('*.json'))
            self.assertEqual(kept, ['0000-native_click_feedback.json', '0001-native_stall_followup.json'])

    def test_entry_inspection_requires_two_actual_reads(self):
        """实际遇敌连续两次相同才能交战斗，不沿用原先点击的目标号。"""
        self.ns['goto'].battle_popup_score.return_value = 1.
        self.ns['battle'].ex_stage.return_value = False
        self.ns['read_stage'] = Mock(side_effect=[(20, 7), (20, 8), (20, 8)])
        self.assertEqual(self.runner.inspect_entry(), (20, 8))
        self.executor.battle.assert_not_called()

    def test_inspection_timeout_closes(self):
        """备战编号读不清时十秒内停止，仍不允许普通开战按钮。"""
        self.ns['goto'].battle_popup_score.return_value = 1.
        self.ns['battle'].ex_stage.return_value = False
        self.ns['read_stage'] = Mock(return_value=None)
        self.ns['time'].monotonic.side_effect = [0., 0., 11.]
        self.runner.close_entry = Mock()
        with self.assertRaises(self.ns['StoryVerificationError']):
            self.runner.inspect_entry()
        self.runner.close_entry.assert_called_once()
        self.executor.battle.assert_not_called()

    def test_input_guard_encounter_does_not_send_another_old_click(self):
        """发送落点前已经遇敌时核对新弹窗，不把旧目标号当作实际关卡或补发落点。"""
        self.ns['read_stage'] = Mock(return_value=(20, 9))
        self.runner.scan = Mock(return_value=(self.field, [self.target]))
        self.session.tap.side_effect = self.ns['TargetTriggered']('entry')
        self.runner.map_frame = Mock(side_effect=[self.field, self.field])
        self.ns['goto'].battle_popup_score.side_effect = [0., 0., 1.]
        self.runner.fight_entry = Mock(return_value=True)
        self.runner.run(ticks=1)
        self.session.tap.assert_called_once()
        self.runner.fight_entry.assert_called_once_with()

    def test_story_is_skipped_before_more_map_input(self):
        """扫描等待遇剧情即调用既有跳过处理，过程中备战校验器始终拒绝开战。"""
        del self.runner.map_frame
        self.ns['battle'].field_visible.side_effect = [False, True]
        self.ns['battle'].battle_state.return_value = dict(state='story')
        seen = []
        self.session.battle_step.side_effect = lambda: seen.append(self.session.entry_validator(self.field))
        self.assertIs(self.runner.map_frame(), self.field)
        self.assertEqual(seen, [False])
        self.assertFalse(self.session.battle_mode)
        self.assertIsNone(self.session.entry_validator)


class MarkerBoundaryTests(unittest.TestCase):
    """紧凑地图下缘的真实任务图标可识别，地图内部相似图案仍不允许点击。"""

    def frame(self, top):
        """以公开模板合成贴近小地图的菱形和左向三角，不使用私人游戏截图。"""
        frame = np.zeros((999, 1776, 3), np.uint8)
        path = Path(__file__).resolve().parents[1] / 'module/campaign/vision/assets/story_marker_tpl.png'
        frame[top:top + 64, 57:121] = cv2.resize(cv2.imread(str(path)), (64, 64))
        cv2.fillConvexPoly(frame, np.array([[35, top + 32], [55, top + 23], [55, top + 41]]), (255, 220, 0))
        return frame

    def test_marker_below_minimap_is_not_masked(self):
        """20-9 失败位置对应的 y=306 位于真实地图下方，不能被多余 HUD 遮罩吞掉。"""
        markers = story_markers(self.frame(306))
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]['kind'], 'edge')

    def test_marker_inside_minimap_remains_masked(self):
        """减少遮罩只到真实地图之外，地图内部图案继续拒绝。"""
        self.assertEqual(story_markers(self.frame(220)), [])

    def test_glowing_low_saturation_outline_keeps_full_glyph_verification(self):
        """青色边框被发光冲淡时仍保留候选；随机同色背景不能替代完整感叹号。"""
        field = self.frame(306)
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
        colored = hsv[:, :, 1] > 130
        hsv[colored, 1] = 90
        field = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.assertEqual(len(story_markers(field)), 1)
        blank = np.zeros_like(field)
        cv2.fillConvexPoly(blank, np.array([[500, 400], [540, 440], [500, 480], [460, 440]]), (240, 200, 150))
        self.assertEqual(story_markers(blank), [])

    def test_white_face_without_blue_context_is_rejected(self):
        """白色内部补候选不单凭感叹号放行；无青色背景时仍拒绝。"""
        field = self.frame(306)
        hsv = cv2.cvtColor(field, cv2.COLOR_BGR2HSV)
        field[hsv[:, :, 1] > 50] = 0
        self.assertEqual(story_markers(field), [])


class EvidenceTests(unittest.TestCase):
    """动作前后图与原始上下文留存有界，后续事件不会覆盖另一个会话。"""

    def test_action_pair_and_capacity(self):
        """一条手势补齐后帧，超过容量后清理该槽旧图；其他槽仍可复盘。"""
        with tempfile.TemporaryDirectory() as folder:
            recorder = ActionEvidence(Path(folder), capacity=2)
            field = np.zeros((20, 20, 3), np.uint8)
            recorder.before('mouse_swipe', field, vector=[20, -40])
            recorder.observed(field)
            self.assertTrue((recorder.folder / '00-after.jpg').exists())
            recorder.record('target', field, arrays={'road': field[:, :, 0]})
            recorder.record('failure', field, arrays={'road': field[:, :, 0]})
            self.assertEqual(len(list(recorder.folder.glob('*.npz'))), 2)
            recorder.record('next', field)
            self.assertEqual(len(list(recorder.folder.glob('*.json'))), 2)
            self.assertFalse((recorder.folder / '00-after.jpg').exists())
            self.assertTrue((recorder.folder / '01-before.jpg').exists())
            self.assertFalse((recorder.folder / '01.npz').exists())

    def test_blocked_decision_frames_survive_slot_wraparound(self):
        """实录22-27：目标旁观察帧被循环槽覆盖；关键决策另存一份不回绕的副本。"""
        with tempfile.TemporaryDirectory() as folder:
            recorder = ActionEvidence(Path(folder), capacity=2)
            field = np.zeros((20, 20, 3), np.uint8)
            recorder.record('goal_view_result', field, state='none')
            for _ in range(3):
                recorder.record('mouse_click', field)
            self.assertTrue((recorder.folder / 'keep' / '0000-goal_view_result.jpg').exists())
            self.assertTrue((recorder.folder / 'keep' / '0000-goal_view_result.json').exists())
            self.assertEqual(len(list((recorder.folder / 'keep').glob('*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
