"""章节完成浮层的输入许可边界；识别替身不连接游戏或模型。"""

import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from module.campaign.vision.chapter_completion import chapter_list_close, chapter_number, chapter_overlay, completion_overlay


class ChapterCompletionTests(unittest.TestCase):
    """完整完成字样、章节关系及唯一控件须同时可靠，缺证据不得产生点击点。"""

    def setUp(self):
        """使用固定UI几何和可控识别结果；地面与人物不参与判定。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.field[375:412, 716:1060] = 220
        cv2.rectangle(self.field, (780, 623), (998, 664), (180, 90, 5), -1)
        self.reader = Mock()
        patcher = patch('module.campaign.vision.chapter_completion.mission_caption_score', return_value=.99)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch('module.campaign.vision.chapter_completion.control_title_prefix',
                        return_value=[784, 624, 911, 662])
        patcher.start()
        self.addCleanup(patcher.stop)

    def results(self, following='CHAPTER21', current_score=.97):
        """按独立三行顺序提供结果，便于检验文字和置信度不能互相补齐。"""
        self.reader.paddleocr.recognize.side_effect = [
            [dict(rec_text='CHAPTER20', rec_score=current_score)],
            [dict(rec_text=following, rec_score=.98)],
        ]

    def test_number_parser_never_truncates_digits_or_ocr_letters(self):
        """CHAPTER前缀与完整数字必须存在，不能把21的尾随l当名称或截成2。"""
        self.assertEqual(chapter_number('CHAPTER20'), 20)
        self.assertEqual(chapter_number('CHAPTER21::巨鲸', caption=True), 21)
        for text in ('20', 'CHAPTER021', 'CHAPTER2l', 'CHAPTER20 CHAPTER21'):
            self.assertIsNone(chapter_number(text), text)

    def test_complete_overlay_returns_the_observed_control(self):
        """只返回实际蓝控件中心和20→21关系，不声明旧章可选内容已经全清。"""
        self.results()
        found = completion_overlay(self.field, self.reader)
        self.assertTrue(found['accepted'])
        self.assertEqual((found['current'], found['following']), (20, 21))
        x, y = found['point']
        left, top, right, bottom = found['box']
        self.assertTrue(left < x < right and top < y < bottom)
        self.assertNotIn('chapter_cleared', found)

    def test_wrong_following_or_low_score_keeps_the_overlay_blocking(self):
        """跳号或低分标题只报告浮层阻塞，不从当前章推算点击许可。"""
        for following, score in (('CHAPTER22', .97), ('CHAPTER21', .89)):
            self.results(following, score)
            found = completion_overlay(self.field, self.reader)
            self.assertFalse(found['accepted'])
            self.assertIsNone(found['point'])

    def test_multiple_controls_never_choose_by_color_or_screen_center(self):
        """两个候选不能擅自挑一个，也不额外读取可能来自不同控件的编号。"""
        self.field[560:750, 470:1305] = 0
        cv2.rectangle(self.field, (660, 623), (860, 664), (180, 90, 5), -1)
        cv2.rectangle(self.field, (900, 623), (1100, 664), (180, 90, 5), -1)
        self.results()
        found = completion_overlay(self.field, self.reader)
        self.assertFalse(found['accepted'])
        self.assertIsNone(found['point'])
        self.reader.paddleocr.recognize.assert_not_called()

    def test_normal_map_never_calls_the_text_model(self):
        """普通地图没有完成白栏时立即返回，避免每次场景观测额外运行OCR。"""
        self.assertIsNone(completion_overlay(np.zeros_like(self.field), self.reader))
        self.reader.paddleocr.recognize.assert_not_called()

    def test_missing_title_separator_never_guesses_a_number_or_calls_ocr(self):
        """缺少完整编号与章名的独立分隔时，保留浮层阻塞而不按当前章推算下一章。"""
        with patch('module.campaign.vision.chapter_completion.control_title_prefix', return_value=None):
            found = completion_overlay(self.field, self.reader)
        self.assertFalse(found['accepted'])
        self.assertIsNone(found['point'])
        self.reader.paddleocr.recognize.assert_not_called()

    def test_unlock_requires_both_complete_chinese_lines(self):
        """UNLOCK不能单独放行；标题与继续提示均须完整且高分，按钮点取自实际文字区域。"""
        cv2.putText(self.field, 'CONTINUE', (850, 635), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        for prompt, score, accepted in [('点击继续。', .99, True), ('击继绣', .99, False),
                                         ('点击继续。', .89, False), ('点击继续。', float('nan'), False)]:
            self.reader.paddleocr.recognize.side_effect = [
                [dict(rec_text='开放探索', rec_score=.99)], [dict(rec_text=prompt, rec_score=score)]]
            with patch('module.campaign.vision.chapter_completion.overlay_anchor',
                       return_value=dict(kind='chapter_unlock', anchor_score=.99)):
                found = chapter_overlay(self.field, self.reader)
            self.assertEqual(found['accepted'], accepted)
            self.assertEqual(found['point'] is not None, accepted)

    def test_chapter_list_close_requires_complete_x_not_plus_or_diagonal(self):
        """列表只能点实际完整X，其他黑色装饰不会因为位于预定区域就获得点击许可。"""
        for shape, accepted in [('x', True), ('plus', False), ('diagonal', False)]:
            field = np.full_like(self.field, 220)
            if shape in ('x', 'diagonal'):
                cv2.line(field, (1089, 108), (1110, 129), (0, 0, 0), 2)
            if shape == 'x':
                cv2.line(field, (1089, 129), (1110, 108), (0, 0, 0), 2)
            if shape == 'plus':
                cv2.line(field, (1089, 119), (1110, 119), (0, 0, 0), 2)
                cv2.line(field, (1100, 108), (1100, 129), (0, 0, 0), 2)
            self.assertEqual(chapter_list_close(field) is not None, accepted, shape)


if __name__ == '__main__':
    unittest.main()
