"""A requested chapter must be verified before campaign movement or combat."""
import ast
import copy
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import re
import sys
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

SOURCE = ast.parse((Path(__file__).resolve().parents[1] / 'module/campaign/task.py').read_text(encoding='utf-8'))


def load(name, **namespace):
    """隔离加载待测方法，避免测试创建 Windows 设备或加载识别模型。"""
    node = next(n for n in ast.walk(SOURCE) if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    namespace.setdefault('pending_chapter', Mock(return_value=None))
    namespace.setdefault('psutil', SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=8*1024**3)))
    namespace.setdefault('read_quest_text', Mock(return_value=None))
    namespace.setdefault('battle_popup_score', Mock(return_value=0.))
    namespace.setdefault('ENTRY_CLOSE_POINT', (1147, 513))
    namespace.setdefault('LOBBY_RETURN_WAIT', 30.)
    namespace.setdefault('return_lobby_quest', Mock(return_value=False))
    namespace.setdefault('destination_lobby_quest', Mock(return_value=False))
    namespace.setdefault('ARK_CHECK', object())
    namespace.setdefault('DESTINATION_SCORE', .8)
    namespace.setdefault('destination_place', Mock(return_value=None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace[name]


def lobby_runner(main_check, **values):
    """组装带独立加载大厅锚点等待的替身，替身sleep不推进真实时间。"""
    runner = SimpleNamespace(device=Mock(), lobby_popup=Mock(return_value=False), **values)
    anchor = load('lobby_anchor', MAIN_CHECK=main_check, logger=Mock(), LOBBY_POPUP_LIMIT=5)
    runner.lobby_anchor = lambda popups: anchor(runner, popups)
    identity = load('quest_identity')
    runner.quest_identity = lambda current: identity(runner, current)
    return runner


class CampaignStartTests(unittest.TestCase):

    def test_recovered_notification_map_does_not_need_a_numbered_lobby_quest(self):
        """解锁后大厅任务可能是返回大厅，已连续确认的原地图可直接续用，不伪造数字任务。"""
        fn = load('ensure_main_map')
        runner = SimpleNamespace(recover_chapter_overlays=Mock(return_value=True), entry_stage=(20, 31),
                                 device=Mock())
        self.assertTrue(fn(runner))
        self.assertIsNone(runner.entry_stage)
        self.assertEqual(runner.device.mock_calls, [])

    def test_notification_recovery_confirms_two_real_map_frames(self):
        """遗留列表经正式会话处理后，连续两帧原章地图才恢复；会话退出始终释放。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        session, win, flow = Mock(), Mock(), Mock()
        session.win.client.return_value = win.client.return_value = field
        flow.handle.side_effect = [True, False, False]
        factory = Mock(return_value=SimpleNamespace(chapter_ocr=Mock()))
        factory.RESOLUTION = (1776, 999)
        fn = load('recover_chapter_overlays', Campaign=factory, CampaignWindow=Mock(return_value=win),
                  CampaignSession=Mock(return_value=session), ChapterOverlayFlow=Mock(return_value=flow),
                  WaitGuard=Mock(), overlay_anchor=Mock(return_value=dict(kind='chapter_list')),
                  chapter_identity=Mock(return_value=(20, 'normal')), cv2=cv2, logger=Mock(),
                  battle=SimpleNamespace(field_visible=Mock(return_value=True)))
        runner = SimpleNamespace(config=SimpleNamespace(PCClient_ScreenNumber=0, PCClient_GameWindowPosition='center'),
                                 device=Mock(image=field))
        self.assertTrue(fn(runner))
        session.start.assert_called_once()
        session.close.assert_called_once()
        self.assertEqual(flow.handle.call_count, 3)
        self.assertEqual(factory.return_value.current_chapter, 20)
        runner.device.click_xy.assert_not_called()

    def test_notification_recovery_rejects_uncertain_background_chapter(self):
        """底层章号读不清时不建立输入会话，不能从浮层或大厅任务猜当前章。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        session, win = Mock(), Mock()
        win.client.return_value = field
        factory = Mock(); factory.RESOLUTION = (1776, 999)
        takeover = type('Takeover', (Exception,), {})
        fn = load('recover_chapter_overlays', Campaign=factory, CampaignWindow=Mock(return_value=win),
                  CampaignSession=session, overlay_anchor=Mock(return_value=dict(kind='chapter_list')),
                  chapter_identity=Mock(side_effect=[(20, 'normal'), None]), cv2=cv2,
                  RequestHumanTakeover=takeover)
        runner = SimpleNamespace(config=SimpleNamespace(PCClient_ScreenNumber=0, PCClient_GameWindowPosition='center'),
                                 device=Mock(image=field))
        with self.assertRaises(takeover):
            fn(runner)
        session.assert_not_called()
        runner.device.click_xy.assert_not_called()

    def test_leftover_entry_popup_is_closed_not_fought(self):
        """实录：上一轮点中敌人后中断，备战弹窗留在地图上；只点关闭，然后照常判断浮层。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        factory = Mock(); factory.RESOLUTION = (1776, 999)
        fn = load('recover_chapter_overlays', Campaign=factory, overlay_anchor=Mock(return_value=None),
                  battle_popup_score=Mock(side_effect=[.99, .2]), cv2=cv2, logger=Mock())
        runner = SimpleNamespace(config=SimpleNamespace(PCClient_ScreenNumber=0, PCClient_GameWindowPosition='center'),
                                 device=Mock(image=field))
        self.assertFalse(fn(runner))
        runner.device.click_xy.assert_called_once_with(1147, 513)
        factory.assert_not_called()

    def test_entry_popup_that_will_not_close_requests_takeover(self):
        """关两次仍在就交人工，不无限点击，也不点进入战斗。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        factory = Mock(); factory.RESOLUTION = (1776, 999)
        takeover = type('Takeover', (Exception,), {})
        fn = load('recover_chapter_overlays', Campaign=factory, overlay_anchor=Mock(return_value=None),
                  battle_popup_score=Mock(return_value=.99), cv2=cv2, logger=Mock(),
                  RequestHumanTakeover=takeover)
        runner = SimpleNamespace(config=SimpleNamespace(PCClient_ScreenNumber=0, PCClient_GameWindowPosition='center'),
                                 device=Mock(image=field))
        with self.assertRaises(takeover):
            fn(runner)
        self.assertEqual(runner.device.click_xy.call_count, 2)

    def test_no_notification_keeps_lobby_path_without_text_models(self):
        """普通界面无锚点时不创建章节OCR/会话，直接交回已有大厅入口。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        factory, reader = Mock(), Mock(); factory.RESOLUTION = (1776, 999)
        fn = load('recover_chapter_overlays', Campaign=factory, overlay_anchor=Mock(return_value=None),
                  chapter_identity=reader, cv2=cv2)
        runner = SimpleNamespace(config=SimpleNamespace(PCClient_ScreenNumber=0, PCClient_GameWindowPosition='center'),
                                 device=Mock(image=field))
        self.assertFalse(fn(runner))
        factory.assert_not_called()
        reader.assert_not_called()

    def test_existing_settlement_is_continued_without_counting_a_win(self):
        """补做入口可清理旧结算；只点现有继续模板，不报告新胜利或开启战斗。"""
        template = object()
        fn = load('recover_settlement', WaitGuard=Mock(), END_FIGHTING=template, logger=Mock())
        runner = SimpleNamespace(device=Mock(), appear=Mock(side_effect=[True, False]),
                                 appear_then_click=Mock(return_value=True))
        fn(runner)
        runner.appear_then_click.assert_called_once_with(template, offset=30, threshold=.85, interval=1)
        self.assertEqual(runner.device.screenshot.call_count, 2)
    """核对输入前的起点与组件保护，防止无效选择触发游戏操作。"""
    def test_missing_runtime_reports_blocked_before_game_actions(self):
        """缺少运行组件时报告待处理，保持设备无操作。"""
        fn = load('runtime_ready', logger=Mock(), report=Mock())
        detector = Mock(side_effect=RuntimeError('Install requirements-yolo.txt'))
        runner = SimpleNamespace(device=Mock())
        with patch.dict(sys.modules, {'module.campaign.vision.detection': SimpleNamespace(detector=detector)}):
            self.assertFalse(fn(runner))
        self.assertEqual(runner.device.mock_calls, [])

    def test_localization_recovery_is_bounded_and_combines_confirmed_wins(self):
        """定位恢复最多两次，累计真实胜利，每次恢复都重新核对选择章节。"""
        takeover = type('Takeover', (Exception,), {})
        campaigns = []
        for count in [2, 1, 0]:
            error = takeover()
            error.__cause__ = RuntimeError('地图定位失败')
            campaigns.append(SimpleNamespace(run=Mock(side_effect=error),
                              state=SimpleNamespace(enemies={i: {'status': 'cleared'} for i in range(count)})))
        reports = []
        factory = Mock(side_effect=campaigns)
        factory.RESOLUTION = (1776, 999)
        fn = load('run', Campaign=factory, RequestHumanTakeover=takeover, GameStuckError=type('Stuck', (Exception,), {}),
                  save_wait_image=Mock(), logger=Mock(), report=lambda *a, **kw: reports.append(a),
                  chapter_number=load('chapter_number', re=re), GOAL_REASONS={})
        runner = SimpleNamespace(config=SimpleNamespace(Client_Platform='win', CampaignMap_MapType='main', CampaignMap_Chapter='current',
                                 PCClient_ScreenNumber=1, PCClient_GameWindowPosition='center', task_delay=Mock()),
                                 device=Mock(), runtime_ready=lambda: True, renew_progress_guard=Mock(), ensure_main_map=Mock(return_value=True),
                                 select_chapter=Mock(return_value=True), cleared_count=load('cleared_count'))
        fn(runner)
        self.assertEqual(factory.call_count, 3)
        self.assertEqual(runner.ensure_main_map.call_count, 3)
        self.assertEqual([c.args[1] for c in runner.select_chapter.call_args_list], [None, None, None])
        self.assertEqual(reports[-1][0], 'partial')
        self.assertIn('3 个敌人', reports[-1][1])
        runner.config.task_delay.assert_called_once_with(schedule=True)
        self.assertEqual(runner.device.target_resolution, (720, 1280))

    def test_localization_recovery_stops_when_reentry_has_no_new_win(self):
        """首次复位后仍无新胜利即停止，避免在大厅与地图间无限循环。"""
        takeover = type('Takeover', (Exception,), {})
        error = takeover()
        error.__cause__ = RuntimeError('地图定位失败')
        campaign = SimpleNamespace(run=Mock(side_effect=error), state=SimpleNamespace(enemies={}))
        factory = Mock(return_value=campaign)
        factory.RESOLUTION = (1776, 999)
        fn = load('run', Campaign=factory, RequestHumanTakeover=takeover, GameStuckError=type('Stuck', (Exception,), {}),
                  save_wait_image=Mock(), logger=Mock(), report=Mock(), chapter_number=load('chapter_number', re=re), GOAL_REASONS={})
        runner = SimpleNamespace(config=SimpleNamespace(Client_Platform='win', CampaignMap_MapType='main', CampaignMap_Chapter='current',
                                 PCClient_ScreenNumber=1, PCClient_GameWindowPosition='center', task_delay=Mock()),
                                 device=Mock(), runtime_ready=lambda: True, renew_progress_guard=Mock(), ensure_main_map=Mock(return_value=True),
                                 select_chapter=Mock(return_value=True), cleared_count=load('cleared_count'))
        fn(runner)
        self.assertEqual(factory.call_count, 2)

    def test_unknown_stop_does_not_trigger_localization_recovery(self):
        """无明确定位故障时保留原始停止结果，不能把通用提示误当成恢复依据。"""
        takeover = type('Takeover', (Exception,), {})
        campaign = SimpleNamespace(run=Mock(side_effect=takeover()), state=SimpleNamespace(enemies={}),
                                   last_result={'state': 'needs_review', 'goal': {'reason': 'no_reachable_target'}})
        factory = Mock(return_value=campaign)
        factory.RESOLUTION = (1776, 999)
        reports = []
        fn = load('run', Campaign=factory, RequestHumanTakeover=takeover, GameStuckError=RuntimeError,
                  save_wait_image=Mock(), logger=Mock(), report=lambda *a: reports.append(a),
                  chapter_number=load('chapter_number', re=re), GOAL_REASONS={'no_reachable_target': '没有可达的目标'})
        runner = SimpleNamespace(config=SimpleNamespace(Client_Platform='win', CampaignMap_MapType='main',
                                 CampaignMap_Chapter='current', PCClient_ScreenNumber=1,
                                 PCClient_GameWindowPosition='center', task_delay=Mock()), device=Mock(),
                                 runtime_ready=lambda: True, renew_progress_guard=Mock(), ensure_main_map=Mock(return_value=True),
                                 select_chapter=Mock(return_value=True), cleared_count=load('cleared_count'))
        fn(runner)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(reports[-1][0], 'blocked')
        self.assertEqual(reports[-1][2], '没有可达的目标')

    def test_critical_memory_pressure_stops_before_loading_detector(self):
        """真实低余量必须先于模型初始化被拦住，保留具体资源原因而非误报模型依赖失败。"""
        reports = []
        fn = load('runtime_ready', psutil=SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=128*1024**2)),
                  report=lambda *a: reports.append(a), logger=Mock())
        detector = Mock()
        configure = Mock()
        with patch.dict(sys.modules, {'module.campaign.vision.detection':
                         SimpleNamespace(configure_inference=configure, detector=detector)}):
            self.assertFalse(fn(SimpleNamespace(config=SimpleNamespace(config_name='test'))))
        configure.assert_not_called()
        detector.assert_not_called()
        self.assertEqual(reports[-1][0], 'blocked')
        self.assertIn('128MiB', reports[-1][2])

    def continuous_runner(self, outcomes, stages):
        """用确认胜利和主线任务序列驱动正式循环，避免依赖真实游戏状态。"""
        takeover = type('Takeover', (Exception,), {})
        campaigns = [SimpleNamespace(run=Mock(side_effect=takeover(reason)),
                     last_result={'state': 'needs_review', 'reason': reason},
                     state=SimpleNamespace(enemies={i: {'status': 'cleared'} for i in range(count)}))
                     for reason, count in outcomes]
        factory = Mock(side_effect=campaigns)
        factory.RESOLUTION = (1776, 999)
        reports = []
        fn = load('run', Campaign=factory, RequestHumanTakeover=takeover, GameStuckError=RuntimeError,
                  save_wait_image=Mock(), logger=Mock(), report=lambda *a: reports.append(a),
                  chapter_number=load('chapter_number', re=re), GOAL_REASONS={})
        runner = SimpleNamespace(config=SimpleNamespace(Client_Platform='win', CampaignMap_MapType='main',
                                 CampaignMap_RunMode='continuous', CampaignMap_Chapter=19, PCClient_ScreenNumber=1,
                                 PCClient_GameWindowPosition='center', task_delay=Mock()), device=Mock(),
                                 runtime_ready=lambda: True, renew_progress_guard=Mock(),
                                 select_chapter=Mock(return_value=True), cleared_count=load('cleared_count'))
        entries = iter(stages)

        def enter():
            """只有重新进入时才更新主线任务，模拟大厅读取的独立证据。"""
            runner.entry_stage = next(entries)
            return True

        runner.ensure_main_map = Mock(side_effect=enter)
        return fn, runner, factory, reports

    def test_continuous_renews_after_progress_and_stops_on_defeat(self):
        """有胜利时越过单段决策与时间预算，战败后立即停止并累加清除数。"""
        fn, runner, factory, reports = self.continuous_runner(
            [('tick_budget', 2), ('segment_timeout', 1), ('battle_defeat', 0)], [(20, 6)] * 3)
        fn(runner)
        self.assertEqual(factory.call_count, 3)
        self.assertEqual(runner.ensure_main_map.call_count, 1)
        self.assertEqual(runner.renew_progress_guard.call_count, 2)
        self.assertTrue(all(c.args[1] is None for c in runner.select_chapter.call_args_list))
        self.assertEqual(reports[-1][0], 'partial')
        self.assertIn('3 个敌人', reports[-1][1])
        self.assertEqual(reports[-1][2], 'battle_defeat')

    def test_continuous_follows_mainline_without_selecting_old_pending_chapter(self):
        """旧章待办不能把连续主线送回过去章节；显式单章模式仍使用指定章节。"""
        fn, runner, factory, _ = self.continuous_runner(
            [('chapter_changed', 1), ('tick_budget', 0)], [(20, 31)])
        fn.__globals__['pending_chapter'] = Mock(return_value=20)
        fn(runner)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual([c.args[1] for c in runner.select_chapter.call_args_list], [None, None])
        fn.__globals__['pending_chapter'].assert_not_called()

    def test_continuous_does_not_renew_empty_segment_or_target_budget(self):
        """无胜利的段预算与单目标输入耗尽都不能作为持续重跑的理由。"""
        for reason, count in [('tick_budget', 0), ('input_budget', 1)]:
            fn, runner, factory, _ = self.continuous_runner([(reason, count)], [(20, 6)])
            fn(runner)
            self.assertEqual(factory.call_count, 1)
            runner.renew_progress_guard.assert_not_called()

    def test_return_lobby_confirms_new_quest_and_follows_current_mainline(self):
        """返回点击不续计时；大厅实际读到新章任务后续跑当前主线，旧章记录不被调用修改。"""
        fn, runner, factory, _ = self.continuous_runner(
            [('story_return_lobby', 0), ('tick_budget', 0)], [None, (21, 1)])
        campaigns = list(factory.side_effect)
        campaigns[0].last_result['chapter'] = 20
        factory.side_effect = campaigns
        fn.__globals__['pending_chapter'] = Mock(return_value=20)
        fn(runner)
        self.assertEqual(runner.ensure_main_map.call_count, 2)
        self.assertEqual([c.args[1] for c in runner.select_chapter.call_args_list], [None, None])
        fn.__globals__['pending_chapter'].assert_not_called()
        runner.renew_progress_guard.assert_called_once()

    def test_return_lobby_without_new_quest_stops_without_renewal(self):
        """大厅仍无法给出新任务时停止，不能把返回动作计为真实进展或反复跳转。"""
        fn, runner, factory, reports = self.continuous_runner([('story_return_lobby', 0)], [None, None])
        campaigns = list(factory.side_effect)
        campaigns[0].last_result['chapter'] = 20
        factory.side_effect = campaigns
        fn(runner)
        self.assertEqual(factory.call_count, 1)
        runner.renew_progress_guard.assert_not_called()
        self.assertIn('尚未确认推进', reports[-1][1])

    def test_continuous_empty_map_checks_quest_once_then_stops(self):
        """无可达目标时只多核对一次大厅任务，任务没前进就结束。"""
        fn, runner, factory, reports = self.continuous_runner([('no_reachable_target', 0)], [(20, 6)] * 2)
        fn(runner)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(runner.ensure_main_map.call_count, 2)
        self.assertIn('20-6', reports[-1][2])
        runner.renew_progress_guard.assert_not_called()

    def test_continuous_advances_to_new_main_chapter(self):
        """主线目标跨章前进后继续最新进度，忽略保留的指定章节设置。"""
        fn, runner, factory, _ = self.continuous_runner(
            [('chapter_changed', 1), ('tick_budget', 0)], [(20, 31), (21, 1)])
        fn(runner)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(runner.ensure_main_map.call_count, 1)
        self.assertEqual(runner.renew_progress_guard.call_count, 1)
        self.assertTrue(all(c.args[1] is None for c in runner.select_chapter.call_args_list))

    def test_chapter_choice_accepts_current_or_supported_positive_number(self):
        """选择限定为当前进度或受支持章节，关卡编号和越界值不能触发输入。"""
        parse = load('chapter_number', re=re)
        self.assertIsNone(parse('current'))
        self.assertEqual(parse('20'), 20)
        for value in ['0', '51', '20-6', '', 'abc']:
            with self.assertRaises(ValueError):
                parse(value)

    def test_chapter_selection_survives_configuration_round_trip(self):
        """数字章节保存后再次读取仍须保留，不能因选项类型不一致退回默认值。"""
        root = Path(__file__).resolve().parents[1]
        spec = json.loads((root / 'module/config/argument/args.json').read_text(encoding='utf-8'))['AutoPush']['CampaignMap']['Chapter']
        source = ast.parse((root / 'module/config/utils.py').read_text(encoding='utf-8'))
        node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_value')
        namespace = {'datetime': datetime}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'parse_value', 'exec'), namespace)
        parse = namespace['parse_value']
        value = parse(20, spec)
        self.assertEqual(value, 20)
        self.assertEqual(parse(json.loads(json.dumps(value)), spec), 20)
        self.assertEqual(parse('current', spec), 'current')

    def chapter_runner(self, identities):
        """构造无真实输入的章节切换环境，超时按一次等待触发。"""
        device = Mock()
        device.image = SimpleNamespace(shape=(999, 1776, 3))
        timer = Mock()
        timer.reached.return_value = True
        reports = []
        fn = load('select_chapter', WaitGuard=Mock(), Timer=Mock(return_value=timer),
                  chapter_identity=Mock(side_effect=identities), cv2=Mock(), logger=Mock(),
                  report=lambda *a, **kw: reports.append(a), GameStuckError=RuntimeError)
        return fn, SimpleNamespace(device=device), reports

    def test_current_progress_and_matching_chapter_do_not_switch(self):
        """当前进度不发送章节点击，指定章节已经选中也不误切下一章。"""
        fn, runner, _ = self.chapter_runner([(20, 'normal')])
        self.assertTrue(fn(runner, Mock(), None))
        self.assertTrue(fn(runner, Mock(), 20))
        runner.device.click_xy.assert_not_called()

    def test_previous_chapter_requires_confirmed_identity_after_click(self):
        """返回上一章后必须读到目标编号才继续，不用点击次数代替结果。"""
        fn, runner, _ = self.chapter_runner([(20, 'normal'), (19, 'normal')])
        # The first post-click frame must be allowed before the fake timer expires.
        fn.__globals__['Timer'].return_value.reached.return_value = False
        self.assertTrue(fn(runner, Mock(), 19))
        runner.device.click_xy.assert_called_once_with(1551, 940)

    def test_locked_next_chapter_stops_without_starting_campaign(self):
        """下一章未解锁时仅点击一次并报告待处理，不在错误章节开战。"""
        fn, runner, reports = self.chapter_runner([(20, 'normal')])
        self.assertFalse(fn(runner, Mock(), 21))
        runner.device.click_xy.assert_called_once_with(1715, 940)
        self.assertEqual(reports[-1][0], 'blocked')
        self.assertIn('21', reports[-1][1])

    def test_only_complete_quest_text_is_accepted(self):
        """主线文字需包含完整关卡，其他任务文字不能作为依据。"""
        for text, expected in [('通关【20-6】', (20, 6)), ('CLEAR 20-6', (20, 6)),
                               ('通关 20', None), ('MOST BEAUTIFUL MAID: END', None)]:
            ocr = Mock(return_value=SimpleNamespace(ocr=lambda _, **kwargs: {'text': text}))
            fn = load('quest_stage', re=re, Ocr=ocr)
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            self.assertEqual(fn(runner), expected)

    def test_lobby_fade_waits_for_matching_fresh_quest_frames(self):
        """首帧空OCR和变化中的任务都不能直接触发入口，须等两次相同有效任务。"""
        clock = Mock(side_effect=[0., .1, .5, .9, 1.3])
        main = object()
        fn = load('confirm_lobby_quest', time=SimpleNamespace(monotonic=clock), MAIN_CHECK=main, logger=Mock())
        runner = lobby_runner(main, appear=Mock(return_value=True),
                              quest_stage=Mock(side_effect=[None, (20, 31), (21, 1), (21, 1)]),
                              lobby_sortie_instruction=False)
        self.assertEqual(fn(runner), ((21, 1), False))
        self.assertEqual(runner.device.screenshot.call_count, 4)
        runner.device.click_xy.assert_not_called()
        runner.appear.return_value = False
        clock.side_effect = [0., .1]
        self.assertIsNone(fn(runner))

    def test_lobby_still_showing_return_waits_for_story_and_new_quest(self):
        """实录21章通关：回大厅后任务仍是“返回大厅”，约十秒后剧情（截图时跳过）结束才出现新任务。"""
        clock = SimpleNamespace(now=0.)
        main = object()
        fn = load('confirm_lobby_quest', time=SimpleNamespace(monotonic=lambda: clock.now), MAIN_CHECK=main,
                  logger=Mock())
        runner = lobby_runner(main, appear=Mock(return_value=True), lobby_sortie_instruction=False)

        def read():
            """前十秒任务栏仍是返回大厅，之后是新章第一关。"""
            runner.lobby_return_pending = clock.now < 10.
            return None if runner.lobby_return_pending else (22, 1)
        runner.quest_stage = Mock(side_effect=read)
        runner.device.sleep.side_effect = lambda seconds: setattr(clock, 'now', clock.now+seconds)
        self.assertEqual(fn(runner), ((22, 1), False))
        clock.now = 0.
        runner.quest_stage = Mock(side_effect=lambda: setattr(runner, 'lobby_return_pending', True))
        self.assertIsNone(fn(runner))
        self.assertLessEqual(clock.now, 30.6)

    def test_fixed_lobby_line_avoids_empty_broad_detection(self):
        """固定单行高分可读取有效主线时，无需再次调用会受淡入干扰的大区域文字检测。"""
        reader = Mock()
        fn = load('quest_stage', re=re, Ocr=Mock(return_value=reader),
                  read_quest_text=Mock(return_value='通关【21-1]'))
        runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
        self.assertEqual(fn(runner), (21, 1))
        reader.ocr.assert_not_called()

    def test_lobby_stability_window_starts_after_cold_ocr_initialization(self):
        """模型首次加载七秒后仍须允许第二次新帧确认，不能把冷启动误作任务消失。"""
        clock = SimpleNamespace(now=0.)
        main = object()
        fn = load('confirm_lobby_quest', time=SimpleNamespace(monotonic=lambda: clock.now), MAIN_CHECK=main,
                  logger=Mock())
        def read():
            """首帧模拟实际OCR冷加载耗时，返回真实可解析任务。"""
            clock.now = max(clock.now, 7.)
            return (21, 1)
        runner = lobby_runner(main, appear=Mock(return_value=True),
                              quest_stage=Mock(side_effect=read), lobby_sortie_instruction=False)
        runner.device.sleep.side_effect = lambda seconds: setattr(clock, 'now', clock.now+seconds)
        self.assertEqual(fn(runner), ((21, 1), False))
        self.assertEqual(runner.device.screenshot.call_count, 2)

    def test_daily_reward_popup_is_closed_before_and_during_lobby_confirmation(self):
        """实录每日奖励浮层先遮住锚点、确认窗口中又淡入：按模板关闭后重新确认，不超过五次。"""
        clock = SimpleNamespace(now=0.)
        main = object()
        fn = load('confirm_lobby_quest', time=SimpleNamespace(monotonic=lambda: clock.now), MAIN_CHECK=main,
                  logger=Mock(), LOBBY_POPUP_LIMIT=5)
        visible = iter([False, True, False, True, True])
        runner = lobby_runner(main, appear=Mock(side_effect=lambda *a, **k: next(visible)),
                              quest_stage=Mock(return_value=(21, 15)), lobby_sortie_instruction=False)
        runner.lobby_popup.side_effect = [True, True]
        runner.device.sleep.side_effect = lambda seconds: setattr(clock, 'now', clock.now+seconds)
        self.assertEqual(fn(runner), ((21, 15), False))
        self.assertEqual(runner.lobby_popup.call_count, 2)
        runner.device.click_xy.assert_not_called()
        runner.appear = Mock(return_value=False)
        runner.lobby_popup = Mock(return_value=True)
        self.assertIsNone(fn(runner))
        self.assertEqual(runner.lobby_popup.call_count, 5)

    def test_sortie_instruction_is_distinct_from_a_numbered_stage(self):
        """完整出击指令独立保存，活动或残缺文字不得虚构下一关，也不接受弱分文本。"""
        for text, accepted in [('前往地面【出击】', True), ('前往地面【出擊】', True), ('【出击】', True), ('出击', False),
                               ('前往地面', False), ('领取地面出击奖励', False), ('通关【21-1】', False)]:
            reader = Mock()
            reader.ocr.return_value = dict(text=text)
            fn = load('quest_stage', re=re, Ocr=Mock(return_value=reader))
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            fn(runner)
            self.assertEqual(runner.lobby_sortie_instruction, accepted)
            reader.ocr.assert_called_once_with(runner.device.image, threshold=.9)

    def test_outpost_quest_is_a_distinct_stable_instruction(self):
        """实录21章通关后大厅任务为前往前哨基地：两帧一致即确认为独立指令，不虚构关卡编号。"""
        clock = SimpleNamespace(now=0.)
        main = object()
        fn = load('confirm_lobby_quest', time=SimpleNamespace(monotonic=lambda: clock.now), MAIN_CHECK=main,
                  logger=Mock())
        runner = lobby_runner(main, appear=Mock(return_value=True), lobby_sortie_instruction=False,
                              lobby_destination_instruction=True, quest_stage=Mock(return_value=None))
        runner.device.sleep.side_effect = lambda seconds: setattr(clock, 'now', clock.now+seconds)
        self.assertEqual(fn(runner), (None, False))

    def test_outpost_visit_clicks_verified_arrow_once_and_waits_for_new_quest(self):
        """只点一次连续核对的任务箭头；进入基地后任务文字更新才算完成，超时不重复点击。"""
        from module.campaign.visible_story import destination_lobby_quest
        for text, accepted in [('前往【前哨基地】', True), ('前往[前哨基地]', True), ('前往 前哨基地', True),
                               ('前往【前哨基地】领取', False), ('前哨基地', False), (None, False),
                               ('前往【方舟】', True), ('前往地面【出击】', False),
                               ('前往【方舟】的【市内】', True), ('前往【方舟】的[简报室]', True),
                               ('前往【方舟】的【市内】领取', False), ('前往【方舟】的', False)]:
            self.assertEqual(destination_lobby_quest(text), accepted)
        clock = SimpleNamespace(now=0.)
        main, outpost = object(), SimpleNamespace(check_button=object())
        point = Mock(return_value=(685.5, 128.))
        fn = load('visit_quest_destination', time=SimpleNamespace(monotonic=lambda: clock.now), MAIN_CHECK=main,
                  page_outpost=outpost, quest_jump_point=point, cv2=SimpleNamespace(cvtColor=lambda x, _: x,
                  COLOR_RGB2BGR=0), logger=Mock(), DESTINATION_QUEST_WAIT=30., LOBBY_JUMP_AREA=(660, 108, 710, 149))
        frames = iter([main, None, outpost.check_button, outpost.check_button])
        runner = SimpleNamespace(device=Mock(), lobby_destination_instruction=True, lobby_quest_text='前往【前哨基地】')
        current = SimpleNamespace(frame=None)
        runner.device.screenshot.side_effect = lambda: setattr(current, 'frame', next(frames))
        runner.appear = lambda button, **_: button is current.frame
        texts = iter(['前往【前哨基地】', '【出击】'])

        def read():
            """先是完成提示中仍为前往前哨基地，之后更新为出击。"""
            runner.lobby_quest_text = next(texts)
            runner.lobby_destination_instruction = runner.lobby_quest_text == '前往【前哨基地】'
        runner.quest_stage = read
        runner.device.sleep.side_effect = lambda seconds: setattr(clock, 'now', clock.now+seconds)
        self.assertTrue(fn(runner))
        runner.device.click_xy.assert_called_once_with(685.5, 128.)
        clock.now = 0.
        frames = iter([main] + [object()]*40)
        runner.device.click_xy.reset_mock()
        self.assertIsNone(fn(runner))
        runner.device.click_xy.assert_called_once()
        self.assertLessEqual(clock.now, 31.5)
        point.return_value = None
        frames = iter([main]*3)
        runner.device.click_xy.reset_mock()
        self.assertFalse(fn(runner))
        runner.device.click_xy.assert_not_called()

    def test_ark_destination_chain_clicks_each_matching_card_once(self):
        """实录23章通关：方舟页卡片地名与任务一致才点，市内→简报室各点一次，任务变为通关后完成。"""
        from module.campaign.visible_story import destination_place
        self.assertEqual(destination_place('前往【方舟】的【市内】'), '市内')
        self.assertEqual(destination_place('前往【前哨基地】'), '前哨基地')
        self.assertIsNone(destination_place('通关【24-1】'))
        clock = SimpleNamespace(now=0.)
        main, ark = object(), object()
        fn = load('visit_quest_destination', time=SimpleNamespace(monotonic=lambda: clock.now), MAIN_CHECK=main,
                  ARK_CHECK=ark, page_outpost=SimpleNamespace(check_button=object()),
                  quest_jump_point=Mock(return_value=(685.5, 128.)), destination_place=destination_place,
                  cv2=SimpleNamespace(cvtColor=lambda x, _: x, COLOR_RGB2BGR=0), logger=Mock(),
                  DESTINATION_QUEST_WAIT=30., LOBBY_JUMP_AREA=(660, 108, 710, 149))
        frames = iter([main, ark, ark, ark, ark, ark])
        runner = SimpleNamespace(device=Mock(), lobby_destination_instruction=True,
                                 lobby_quest_text='前往【方舟】的【市内】')
        current = SimpleNamespace(frame=None)
        runner.device.screenshot.side_effect = lambda: setattr(current, 'frame', next(frames))
        runner.appear = lambda button, **_: button is current.frame
        texts = iter(['前往【方舟】的【市内】', '前往【方舟】的【市内】', '前往【方舟】的【简报室】',
                      '前往【方舟】的【简报室】', '通关【24-1】'])

        def read():
            """点卡片后任务依次推进，加载期间同一任务会重复读到。"""
            runner.lobby_quest_text = next(texts)
            runner.lobby_destination_instruction = runner.lobby_quest_text.startswith('前往')
        runner.quest_stage = read
        runner.ark_destination_card = Mock(return_value=(347., 628.))
        runner.device.sleep.side_effect = lambda seconds: setattr(clock, 'now', clock.now+seconds)
        self.assertTrue(fn(runner))
        self.assertEqual([c.args for c in runner.device.click_xy.call_args_list],
                         [(685.5, 128.), (347., 628.), (347., 628.)])
        self.assertEqual([c.args[0] for c in runner.ark_destination_card.call_args_list], ['市内', '简报室'])

    def test_ark_card_requires_two_frames_with_the_quest_place(self):
        """卡片地名两帧都等于任务地名才返回中心；读到别的地名或离开方舟页不点。"""
        ark = object()
        fn = load('ark_destination_card', ARK_CHECK=ark, ARK_DESTINATION_LABEL=(292, 600, 402, 656), re=re)
        for texts, expected in [(['市内', '市内'], (347., 628.)), (['市内', '简报室'], None), (['模拟室'], None)]:
            reader = Mock()
            reader.ocr.side_effect = [dict(text=text) for text in texts]
            runner = SimpleNamespace(device=Mock(), config=SimpleNamespace(Optimization_OcrModelType='mobile'),
                                     appear=lambda button, **_: button is ark)
            fn.__globals__['Ocr'] = Mock(return_value=reader)
            self.assertEqual(fn(runner, '市内'), expected)


if __name__ == '__main__':
    unittest.main()
