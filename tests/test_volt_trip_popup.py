"""每天首次登录弹出的 VOLT TRIP 小游戏要在登录流程里关掉，不能干等到超时重开游戏。"""
import ast
from pathlib import Path
import unittest

from tests.test_login_download_progress import FakeGuard, Runner, Stuck


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    cls = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name][0]
    body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), namespace)
    return namespace[method_name]


def asset_areas(path):
    areas = {}
    for node in ast.parse((ROOT / path).read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            kwargs = {k.arg: k.value for k in node.value.keywords}
            areas[node.targets[0].id] = ast.literal_eval(kwargs['area'])['zh-CN']
    return areas


class VoltTripRunner(Runner):
    def __init__(self, handles_popup):
        super().__init__([])
        self.handles_popup = handles_popup
        self.closed = False

    def screenshot(self):
        self.frame = 'main' if self.closed else 'volt_trip'

    def handle_volt_trip(self):
        if self.handles_popup and self.frame == 'volt_trip':
            self.closed = True
            return True
        return False


class VoltTripLoginTest(unittest.TestCase):
    def setUp(self):
        FakeGuard.instances = []

    def test_login_closes_volt_trip(self):
        VoltTripRunner(handles_popup=True)._handle_app_login()

    def test_unhandled_popup_times_out(self):
        with self.assertRaises(Stuck):
            VoltTripRunner(handles_popup=False)._handle_app_login()


class VoltTripHandlerTest(unittest.TestCase):
    handle = staticmethod(load_method('module/handler/info_handle.py', 'InfoHandler', 'handle_volt_trip',
                                      {'VOLT_TRIP_CHECK': 'check', 'VOLT_TRIP_CLOSE': 'close'}))

    class Fake:
        def __init__(self, visible):
            self.visible = set(visible)
            self.clicked = []

        def appear(self, button, **kwargs):
            return button in self.visible

        def appear_then_click(self, button, **kwargs):
            if button in self.visible:
                self.clicked.append(button)
                return True
            return False

    def test_clicks_close_only_on_volt_trip(self):
        popup = self.Fake(['check', 'close'])
        self.assertTrue(self.handle(popup))
        self.assertEqual(popup.clicked, ['close'])

        # 别的页面右上角也有 X，没有 VOLT TRIP 标题时不点
        other = self.Fake(['close'])
        self.assertFalse(self.handle(other))
        self.assertEqual(other.clicked, [])

    def test_assets(self):
        areas = asset_areas('module/handler/assets.py')
        x1, y1, x2, y2 = areas['VOLT_TRIP_CLOSE']
        self.assertGreater(x1, 600)
        self.assertLess(y2, 200)
        for name in ('VOLT_TRIP_CHECK', 'VOLT_TRIP_CLOSE'):
            self.assertTrue((ROOT / f'assets/zh-CN/handler/{name}.png').exists(), name)


if __name__ == '__main__':
    unittest.main()
