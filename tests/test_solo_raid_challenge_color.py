"""单人突击「挑战」按钮：新版发光按钮要能点，灰掉的按钮仍要认成不可点。"""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_SOURCE = (ROOT / 'module/solo_raid/challenge.py').read_text(encoding='utf-8')
RAID_SOURCE = (ROOT / 'module/solo_raid/solo_raid.py').read_text(encoding='utf-8')
ASSET_COLOR = (246, 179, 167)


def tolerance(color1, color2):
    # 与 module/base/utils.color_similar 相同的 Photoshop 容差公式（该模块依赖 cv2，这里不直接导入）
    diff = [int(a) - int(b) for a, b in zip(color1, color2)]
    return max(max(diff), 0) - min(min(diff), 0)


def threshold_value():
    tree = ast.parse(CHALLENGE_SOURCE)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'CHALLENGE_COLOR_THRESHOLD' for t in node.targets
        ):
            return node.value.value
    raise AssertionError('CHALLENGE_COLOR_THRESHOLD not defined')


def challenge_thresholds(source):
    """所有 appear/appear_then_click(CHALLENGE, threshold=...) 的阈值写法。"""
    found = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ('appear', 'appear_then_click')
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == 'CHALLENGE'
        ):
            for kw in node.keywords:
                if kw.arg == 'threshold':
                    found.append(ast.unparse(kw.value))
    return found


class ChallengeColorTest(unittest.TestCase):
    def test_new_glowing_button_is_clickable(self):
        # 实录 2026-11-12 日志里按钮平均色，容差 11～18；原阈值 10 一次都没过
        for color in [(245, 171, 156), (247, 173, 156), (248, 174, 157), (246, 161, 149)]:
            self.assertLessEqual(tolerance(color, ASSET_COLOR), threshold_value(), color)

    def test_grey_button_still_rejected(self):
        for color in [(120, 119, 119), (150, 150, 150), (90, 90, 92)]:
            self.assertGreater(tolerance(color, ASSET_COLOR), threshold_value(), color)

    def test_every_challenge_check_uses_shared_threshold(self):
        values = challenge_thresholds(CHALLENGE_SOURCE) + challenge_thresholds(RAID_SOURCE)
        self.assertEqual(len(values), 3)
        self.assertEqual(set(values), {'CHALLENGE_COLOR_THRESHOLD'})


if __name__ == '__main__':
    unittest.main()
