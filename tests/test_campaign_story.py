"""剧情优先级的输入边界：横竖屏、确认框、防抖及长于四帧的等待。"""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import math
import cv2
import numpy as np
from unittest.mock import Mock, patch


def method(path, name, **namespace):
    """隔离加载方法，测试不创建 Windows 窗口、模型或真实输入通道。"""
    source = ast.parse((Path(__file__).resolve().parents[1] / path).read_text(encoding='utf-8'))
    node = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace[name]


class StoryPriorityTests(unittest.TestCase):
    """只对已确认的剧情控件发输入，不让跳过确认跨越到普通弹窗。"""
    def setUp(self):
        """各测试共用固定时钟及模板替身，输入调用均留在 Mock 中。"""
        self.skip = Mock()
        self.skip.match.return_value = False
        self.story_skip = Mock(return_value=(0., None))
        self.ocr = Mock()
        self.ocr.return_value.ocr.return_value = {'text': '', 'details': []}
        self.device = SimpleNamespace(image=SimpleNamespace(shape=(999, 1776, 3)), click=Mock(),
                                      click_xy=Mock(), config=SimpleNamespace(Optimization_OcrModelType='mobile'))
        self.fn = method('module/handler/story_priority.py', 'skip_story_frame',
                         time=SimpleNamespace(monotonic=lambda: 10), logger=Mock(), cv2=Mock(),
                         portrait_story_menu=Mock(return_value=False))
        self.modules = patch.dict(sys.modules, {
            'module.ui.assets': SimpleNamespace(SKIP=self.skip),
            'module.campaign.vision.battle': SimpleNamespace(story_skip=self.story_skip),
            'module.ocr.ocr': SimpleNamespace(Ocr=self.ocr),
            'module.handler.assets': SimpleNamespace(CONFIRM_A=self.skip, CONFIRM_B=self.skip, CONFIRM_C=self.skip)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_landscape_skip_uses_current_template_point_and_debounces(self):
        """横屏使用本帧模板落点，防抖期间仍要求重拍且不重复点击。"""
        self.story_skip.return_value = (.99, [1690.3, 30.8])
        self.assertTrue(self.fn(self.device))
        self.assertTrue(self.fn(self.device))
        self.device.click_xy.assert_called_once_with(1690, 31)
        self.device.click.assert_not_called()

    def test_portrait_keeps_existing_skip_template(self):
        """日常竖屏继续使用原模板，不能套用主线横屏坐标。"""
        self.device.image.shape = (1280, 720, 3)
        self.skip.match.return_value = True
        self.assertTrue(self.fn(self.device))
        self.device.click.assert_called_once_with(self.skip)
        self.device.click_xy.assert_not_called()

    def test_landscape_skip_confirmation_requires_skip_context(self):
        """横屏确认框必须属于刚识别的跳过操作，点击 OCR 确认文字中心。"""
        self.device._story_skip_pending = 18
        self.story_skip.return_value = (.99, [1690, 31])
        self.ocr.return_value.ocr.return_value = {'text': '是否跳过剧情 确认',
                                                'details': [{'text': '确认', 'bbox': [300, 200, 340, 220]}]}
        self.assertTrue(self.fn(self.device))
        self.device.click_xy.assert_called_once_with(848, 460)
        self.assertEqual(self.device._story_skip_pending, 0)

    def test_glowing_portrait_requires_four_words_and_two_frames(self):
        """低模板提案不能单独点击，四词核验还须连续两帧同一实际控件。"""
        self.device.image.shape = (1280, 720, 3)
        self.skip.match.side_effect = [False, True, False, True]
        self.skip.button = (616, 22, 671, 41)
        self.fn.__globals__['portrait_story_menu'].return_value = True
        self.assertTrue(self.fn(self.device))
        self.device.click.assert_not_called()
        self.assertTrue(self.fn(self.device))
        self.device.click.assert_called_once_with(self.skip)

    def test_portrait_menu_rejects_partial_words_and_low_or_invalid_scores(self):
        """缺失菜单、低分或NaN不能把普通顶部控件当作剧情跳过。"""
        fn = method('module/handler/story_priority.py', 'portrait_story_menu', math=math, cv2=cv2)
        image = np.zeros((1280, 720, 3), np.uint8)
        reader = Mock()
        rows = [dict(rec_text=word, rec_score=.99) for word in ('HIDE', 'AUTO', 'LOG', 'SKIP')]
        reader.paddleocr.recognize.return_value = rows
        self.assertTrue(fn(image, reader))
        for text, score in [('SKI', .99), ('SKIP', .94), ('SKIP', float('nan'))]:
            rows[-1] = dict(rec_text=text, rec_score=score)
            self.assertFalse(fn(image, reader))

    def test_unrelated_or_expired_confirmation_is_never_clicked(self):
        """购买等普通确认框及过期跳过上下文都不发送输入。"""
        self.device._story_skip_pending = 18
        self.ocr.return_value.ocr.return_value = {'text': '是否购买 确认',
                                                'details': [{'text': '确认', 'bbox': [300, 200, 340, 220]}]}
        self.assertFalse(self.fn(self.device))
        self.device._story_skip_pending = 9
        self.assertFalse(self.fn(self.device))
        self.device.click_xy.assert_not_called()

    def test_screenshot_waits_past_four_frames_and_retains_timeout(self):
        """剧情处理超过四帧时仍不暴露给任务，并持续检查独立等待上限。"""
        skip = Mock(side_effect=[True] * 6 + [False])
        guard = Mock()
        capture = Mock()
        fn = method('module/device/win/device.py', 'screenshot',
                    super=lambda: SimpleNamespace(screenshot=capture), publish_preview_frame=Mock())
        image = object()
        device = SimpleNamespace(current_window=SimpleNamespace(image=image),
                                 config=SimpleNamespace(stop_event=None), sleep=Mock(), stuck_record_check=Mock())
        with patch.dict(sys.modules, {'module.handler.story_priority': SimpleNamespace(skip_story_frame=skip),
                                      'module.base.watchdog': SimpleNamespace(WaitGuard=Mock(return_value=guard))}):
            self.assertIs(fn(device), image)
        self.assertEqual(capture.call_count, 7)
        self.assertEqual(guard.check.call_count, 6)


if __name__ == '__main__':
    unittest.main()
