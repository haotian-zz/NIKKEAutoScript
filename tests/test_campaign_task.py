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

SOURCE = ast.parse((Path(__file__).resolve().parents[1] / 'module/campaign/task.py').read_text(encoding='utf-8'))


def load(name, **namespace):
    """隔离加载待测方法，避免测试创建 Windows 设备或加载识别模型。"""
    node = next(n for n in ast.walk(SOURCE) if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(node)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace[name]


class CampaignStartTests(unittest.TestCase):
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
                                 device=Mock(), runtime_ready=lambda: True, ensure_main_map=Mock(return_value=True),
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
                                 device=Mock(), runtime_ready=lambda: True, ensure_main_map=Mock(return_value=True),
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
                                 runtime_ready=lambda: True, ensure_main_map=Mock(return_value=True),
                                 select_chapter=Mock(return_value=True), cleared_count=load('cleared_count'))
        fn(runner)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(reports[-1][0], 'blocked')
        self.assertEqual(reports[-1][2], '没有可达的目标')

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
            ocr = Mock(return_value=SimpleNamespace(ocr=lambda _: {'text': text}))
            fn = load('quest_stage', re=re, Ocr=ocr)
            runner = SimpleNamespace(config=SimpleNamespace(Optimization_OcrModelType='mobile'), device=Mock())
            self.assertEqual(fn(runner), expected)


if __name__ == '__main__':
    unittest.main()
