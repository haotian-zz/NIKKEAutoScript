"""活动某一章没有困难难度素材时（MARIAn 的 Story 1），进普通难度页面不能因缺素材报错。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


source = (Path(__file__).resolve().parents[1] / 'module/event/story.py').read_text(encoding='utf-8')
cls = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)][0]
body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'open_story_page']


class FakeTimer:
    def __init__(self, *args):
        pass

    def start(self):
        return self

    def reset(self):
        pass

    def reached(self):
        return False


namespace = {'Timer': FakeTimer, 'FIGHT_CLOSE': 'close', 'logger': SimpleNamespace(info=lambda *a: None),
             'WaitGuard': lambda *a, **k: SimpleNamespace(check=lambda: None)}
exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'story_page', 'exec'), namespace)


class Runner:
    open_story_page = namespace['open_story_page']

    def __init__(self):
        # 只有 Story 1 普通难度和章节入口，没有 STORY_1_HARD / STORY_1_HARD_*。
        self.event_assets = SimpleNamespace(STORY_STAGE_CHECK='panel', EVENT_GOTO_STORY_1='goto1',
                                            STORY_1_NORMAL='normal1', STORY_1_CHECK='check1')
        self.event = SimpleNamespace(type=1)
        self.state = 'event'
        self.clicks = []
        self.device = SimpleNamespace(screenshot=lambda: None)

    def back_to_event(self):
        self.state = 'event'

    def appear_story_part(self, story):
        return self.state == 'list'

    def appear(self, button, **kwargs):
        return (button == 'goto1' and self.state == 'event') or (button == 'normal1' and self.state == 'story')

    def appear_then_click(self, button, **kwargs):
        if not self.appear(button):
            return False
        self.clicks.append(button)
        self.state = {'goto1': 'story', 'normal1': 'list'}[button]
        return True


class StoryPageMissingHardAssetTest(unittest.TestCase):
    def test_normal_only_part_opens_without_hard_asset(self):
        runner = Runner()
        self.assertTrue(runner.open_story_page('story_1_normal'))
        self.assertEqual(runner.clicks, ['goto1', 'normal1'])


if __name__ == '__main__':
    unittest.main()
