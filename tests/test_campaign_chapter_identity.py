"""固定章节字段快路径的严格身份边界与原流水线回退；不加载模型或连接游戏。"""

import unittest
from unittest.mock import Mock

import numpy as np

from module.campaign.vision.chapter import chapter_identity


class ChapterIdentityTests(unittest.TestCase):
    """只给完整高分单行字段输入许可，未知结果继续遵守原流水线与完整字符串解析。"""

    def setUp(self):
        """使用假识别器与空客户区；所有文字明确由替身返回，不从测试期望反填。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.reader = Mock()
        self.reader.ocr.side_effect = lambda *args, **kwargs: dict(text='')

    def results(self, number='20', difficulty='NORMAL', number_score=.98, difficulty_score=.97):
        """提供两行直接识别结果，便于独立检验章号与难度的接受条件。"""
        self.reader.paddleocr.recognize.return_value = [
            dict(rec_text=number, rec_score=number_score), dict(rec_text=difficulty, rec_score=difficulty_score)]

    def test_complete_normal_and_hard_bypass_text_detection(self):
        """完整数字及NORMAL/HARD可直接返回；两种难度分别解析，不借用当前任务身份。"""
        for number, difficulty, expected in [('20', 'NORMAL', (20, 'normal')), ('41', 'HARD', (41, 'hard'))]:
            self.results(number, difficulty)
            self.assertEqual(chapter_identity(self.field, self.reader), expected)
        self.reader.ocr.assert_not_called()

    def test_partial_or_ambiguous_text_does_not_gain_fast_acceptance(self):
        """额外字符、分支号、字母替代数字或未知难度不能在快路径补全或截断。"""
        for number, difficulty in [('20-31', 'NORMAL'), ('2O', 'NORMAL'), ('0', 'NORMAL'),
                                   ('200', 'NORMAL'), ('20', 'NORMAL extra'), ('20', '')]:
            self.results(number, difficulty)
            self.assertIsNone(chapter_identity(self.field, self.reader))

    def test_dimmed_hard_tag_rereads_without_list_icon(self):
        """实录：通关浮层下 HARD 标签变暗，左侧图标被读成多一个字；去掉图标补读，只接受 HARD。"""
        self.results('04', '目HARD', difficulty_score=.869)
        texts = iter(['04', '目HARD', 'HARD'])
        self.reader.ocr.side_effect = lambda *args, **kwargs: dict(text=next(texts))
        self.assertEqual(chapter_identity(self.field, self.reader), (4, 'hard'))
        texts = iter(['04', '目HARD', 'NORMAL'])
        self.assertIsNone(chapter_identity(self.field, self.reader))

    def test_low_or_nonfinite_score_uses_unchanged_legacy_thresholds(self):
        """低分或非有限分数回退原流程，章号0.95及难度0.9均不放宽。"""
        for scores in [(.94, .97), (.98, .89), (float('nan'), .97), (.98, float('inf'))]:
            self.reader.reset_mock()
            self.results(number_score=scores[0], difficulty_score=scores[1])
            self.reader.ocr.side_effect = [dict(text='20'), dict(text='NORMAL')]
            self.assertEqual(chapter_identity(self.field, self.reader), (20, 'normal'))
            self.assertEqual([call.kwargs['threshold'] for call in self.reader.ocr.call_args_list], [.95, .9])

    def test_invalid_geometry_is_rejected_before_models(self):
        """未知分辨率先拒绝，不对错位字段运行模型或产生身份。"""
        for field in [None, np.zeros((1280, 720, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                chapter_identity(field, self.reader)
        self.reader.paddleocr.recognize.assert_not_called()
        self.reader.ocr.assert_not_called()


if __name__ == '__main__':
    unittest.main()
