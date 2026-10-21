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
                     goto=SimpleNamespace(battle_popup_score=Mock(return_value=0.), VIEWPORT_PIXEL=(888., 500.)),
                     battle=Mock(), ClearanceJournal=Mock(), SceneEnemyMemory=Mock(),
                     stable_candidates=Mock(), ChapterSweep=Mock(),
                     SweepUnconfirmed=type('SweepUnconfirmed', (RuntimeError,),
                                           {'reason': 'chapter_clearance_unconfirmed'}),
                     squad_arrow=Mock(return_value=None), squad_offscreen=Mock(return_value=None),
                     story_markers=story_markers, pan_scene=Mock())
    from module.campaign.vision.native_motion import ClickProgress, NativeMotion, combine_motion
    namespace.update(ClickProgress=ClickProgress, NativeMotion=NativeMotion, combine_motion=combine_motion,
                     MinimapStability=Mock(return_value=SimpleNamespace(observe=lambda field: dict(state='unknown'))))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
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
        self.reveal = Mock()
        patcher = patch.dict(sys.modules, {'module.campaign.navigation.story_navigation':
                                         SimpleNamespace(reveal_story_marker=self.reveal)})
        patcher.start()
        self.addCleanup(patcher.stop)

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

    def test_entry_title_requires_complete_anchor_and_preserves_branches(self):
        """Boss标签不当分支B，普通分支仍保留全号，EX/残缺/多编号均拒绝。"""
        parse = self.ns['parse_entry_title']
        for text, expected in [('20-31BOSS', (20, 31)), ('20-31 B0SS', (20, 31)),
                               ('20-23STAGE', (20, 23)), ('20-6B-1STAGE', (20, 6, 'B', 1))]:
            self.assertEqual(parse(text), expected)
        for text in ('20-31', '20-6BSTAGE', '20-6B-STAGE', 'EX STAGE', 'EX 20-6 STAGE',
                     '20-6STAGE20-7', '20-31BOSS extra', '0-31BOSS'):
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

    def test_empty_view_requires_full_sweep_before_advancing(self):
        """视野没有怪时启动全章巡检；零可见敌人不能直接作为换章理由。"""
        self.ns['read_stage'] = Mock(return_value=(20, 6))
        self.runner.scan = Mock(return_value=(self.field, []))
        proof = dict(state='verified_empty')
        self.ns['ChapterSweep'].return_value.run.return_value = dict(state='verified_empty', proof=proof)
        self.runner.advance_chapter = Mock(return_value=dict(state='continue', reason='chapter_changed'))
        self.assertEqual(self.runner.run(ticks=1)['reason'], 'chapter_changed')
        self.runner.advance_chapter.assert_called_once_with(proof)
        self.session.tap.assert_not_called()

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

    def test_empty_view_can_request_current_quest_once_without_fighting(self):
        """定位任务只提供找方向辅助；同一任务不重复点击，也不能直接开战或换章。"""
        self.runner.quest = (20, 18)
        self.ns['quest_jump_point'] = Mock(return_value=(1749., 100.))
        self.ns['read_stage'] = Mock(return_value=(20, 18))
        self.assertTrue(self.runner.guide_quest(self.field))
        self.assertFalse(self.runner.guide_quest(self.field))
        self.session.tap.assert_called_once_with((1749., 100.), interval=2.)
        self.executor.battle.assert_not_called()
        self.assertIsNone(self.runner.memory.previous)

    def test_quest_change_between_frames_rejects_locator(self):
        """任务或控件在两帧间变化时不发送旧定位指令。"""
        self.runner.quest = (20, 18)
        self.ns['quest_jump_point'] = Mock(return_value=(1749., 100.))
        self.ns['read_stage'] = Mock(return_value=(20, 21))
        self.assertFalse(self.runner.guide_quest(self.field))
        self.session.tap.assert_not_called()

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
        self.runner.marker_pans = 4
        self.assertFalse(self.runner.reveal_marked_area(self.field))

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
        self.runner.marker_pans = 0
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

    def test_unverified_chapter_cannot_advance(self):
        """任务已经进入下一章也必须等待当前章清怪证明。"""
        self.runner.quest = (21, 1)
        with self.assertRaises(self.ns['SweepUnconfirmed']):
            self.runner.advance_chapter(dict(state='incomplete', pending=[dict(kind='enemy')]))
        self.session.tap.assert_not_called()

    def test_verified_empty_requires_completed_story_and_new_identity(self):
        """本章清空、主线已推进和下一章连续两帧身份缺一不可。"""
        self.runner.quest = (21, 1)
        self.ns['chapter_identity'].return_value = (21, 'normal')
        result = self.runner.advance_chapter(dict(state='verified_empty'))
        self.assertEqual(result['reason'], 'chapter_changed')
        self.session.tap.assert_called_once_with((1715, 940), interval=2.)
        self.assertEqual(self.ns['chapter_identity'].call_count, 2)

    def test_automatic_transition_returns_to_uncleared_chapter(self):
        """游戏自动前进时先返回旧章补清怪，不能据此跳过旧章巡检。"""
        self.ns['chapter_identity'].return_value = (21, 'normal')
        self.runner.restore_chapter = Mock()
        self.runner.run(ticks=1)
        self.runner.restore_chapter.assert_called_once_with((21, 'normal'))
        self.runner.memory.observe.assert_not_called()

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

    def test_unobserved_click_stops_without_facility_input(self):
        """一直缺失位移观测时保留unknown，不能请求设施交互或记录unreachable。"""
        clock = [0.]
        self.ns['time'].monotonic.side_effect = lambda: clock[0]
        self.session.pause.side_effect = lambda seconds: clock.__setitem__(0, clock[0]+seconds)
        self.ns['NativeMotion'] = Mock()
        self.ns['NativeMotion'].return_value.observe.return_value = dict(state='unknown', reason='squad_unobserved')
        with self.assertRaises(self.ns['StoryVerificationError']) as raised:
            self.runner.wait_entry(self.field)
        self.assertEqual(raised.exception.reason, 'native_observation_unknown')
        self.runner.memory.identify.assert_not_called()
        self.session.tap.assert_not_called()

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


if __name__ == '__main__':
    unittest.main()
