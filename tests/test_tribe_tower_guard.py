"""企业塔逐塔推进时重置任务总时限，真卡住时仍由原时限停止；不加载模型或连接游戏。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


source = ast.parse((Path(__file__).resolve().parents[1] / 'module/tribe_tower/tribe_tower.py').read_text(encoding='utf-8'))
cls = next(node for node in ast.walk(source) if isinstance(node, ast.ClassDef) and node.name == 'TribeTower')
methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


class NoOpportunityRemain(Exception):
    pass


def load(*names, **ns):
    ns.setdefault('NoOpportunityRemain', NoOpportunityRemain)
    ns.setdefault('report', Mock())
    ns.setdefault('logger', Mock())
    body = [methods[name] for name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), 'tribe_tower', 'exec'), ns)
    return ns


class TribeTowerGuardTests(unittest.TestCase):
    def test_each_tower_renews_execution_guard(self):
        """实录 2026-10-24：五座塔合计超过 900 秒；每换一座塔都算真实进展。"""
        ns = load('renew_progress_guard', '_run')
        guard = Mock()
        handler = SimpleNamespace(device=Mock(_execution_guard=guard),
                                  available_company=[{'name': 'ELYSION'}, {'name': 'MISSILIS'}, {'name': 'TETRA'}],
                                  config=SimpleNamespace(Overcome_OnlyToCompleteDailyMission=False),
                                  ensure_into_tribe_tower=Mock(side_effect=NoOpportunityRemain),
                                  ensure_back=Mock())
        handler.renew_progress_guard = lambda: ns['renew_progress_guard'](handler)
        ns['_run'](handler)
        self.assertEqual(guard.progress.call_count, 3)
        self.assertEqual(handler.available_company, [])

    def test_missing_guard_is_ignored(self):
        """调试或单测里没有任务总时限时不报错。"""
        ns = load('renew_progress_guard')
        ns['renew_progress_guard'](SimpleNamespace(device=SimpleNamespace()))


if __name__ == '__main__':
    unittest.main()
