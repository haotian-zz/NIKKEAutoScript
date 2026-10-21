"""章节浮层闭环的跨帧与一次输入保护，不连接游戏或模型。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from module.campaign.chapter_flow import ChapterOverlayFlow, ChapterOverlayUnconfirmed


class ChapterFlowTests(unittest.TestCase):
    """实际章号、同一控件与最终新帧三者缺一时，不允许浮层点击。"""

    def setUp(self):
        """固定时钟与假会话；所有输入只记到Mock，避免测试产生真实点击。"""
        self.field = np.zeros((999, 1776, 3), np.uint8)
        self.session = Mock(expected_identity=(20, 'normal'))
        self.session.win.client.return_value = self.field
        self.flow = ChapterOverlayFlow(self.session, Mock())
        self.evidence = dict(kind='chapter_completion', accepted=True, current=20, following=21, point=[880., 640.])
        self.clock = Mock(return_value=0.)
        for target, value in [('time', SimpleNamespace(monotonic=self.clock)),
                              ('overlay_anchor', Mock(return_value=dict(kind='chapter_completion'))),
                              ('chapter_overlay', Mock(side_effect=lambda *args: dict(self.evidence))),
                              ('chapter_identity', Mock(return_value=(20, 'normal')))]:
            patcher = patch('module.campaign.chapter_flow.'+target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_final_guard_rejects_a_different_overlay(self):
        """两帧确认后弹窗换了，最终输入校验必须拒绝，不能拿旧控件坐标继续。"""
        self.flow.handle(self.field)
        point, guard = self.session.tap_chapter_overlay.call_args.args
        self.assertEqual(point, [880., 640.])
        self.assertTrue(guard(self.field))
        self.evidence.update(kind='chapter_unlock', point=[883., 632.])
        self.assertFalse(guard(self.field))

    def test_unresponsive_overlay_is_not_clicked_twice(self):
        """同类浮层点击后等待响应，八秒仍存在则停，不重复点击或伪造进展。"""
        self.assertTrue(self.flow.handle(self.field))
        self.clock.return_value = 3.
        self.assertTrue(self.flow.handle(self.field))
        self.clock.return_value = 9.
        with self.assertRaises(ChapterOverlayUnconfirmed):
            self.flow.handle(self.field)
        self.session.tap_chapter_overlay.assert_called_once()

    def test_consecutive_unlock_titles_are_distinct_and_final_guard_is_specific(self):
        """开放探索后出现快速模拟解锁可继续；上一条提示的最终点击许可不能跨正文使用。"""
        self.evidence = dict(kind='chapter_unlock', accepted=True, title='开放探索', prompt='点击继续。',
                             point=[883., 632.])
        self.flow.handle(self.field)
        _, previous_guard = self.session.tap_chapter_overlay.call_args.args
        self.evidence['title'] = '解锁快速模拟立即完成'
        self.assertFalse(previous_guard(self.field))
        self.flow.handle(self.field)
        self.assertEqual(self.session.tap_chapter_overlay.call_count, 2)
        self.assertEqual(len(self.flow.sent), 2)

    def test_wrong_chapter_or_uncertain_text_has_no_input(self):
        """实际章号不匹配或文字未确认时，保留阻塞直至超时，绝不从期望章补齐。"""
        for field_identity, accepted in [((21, 'normal'), True), ((20, 'normal'), False)]:
            with self.subTest(identity=field_identity, accepted=accepted):
                self.flow.unconfirmed_since = None
                self.clock.return_value = 0.
                self.evidence['accepted'] = accepted
                with patch('module.campaign.chapter_flow.chapter_identity', return_value=field_identity):
                    self.flow.handle(self.field)
                    self.clock.return_value = 11.
                    with self.assertRaises(ChapterOverlayUnconfirmed):
                        self.flow.handle(self.field)
        self.session.tap_chapter_overlay.assert_not_called()

    def test_normal_map_passes_without_ocr_or_input(self):
        """没有固定浮层锚点时交还地图流程，额外文字识别与输入均不运行。"""
        with patch('module.campaign.chapter_flow.overlay_anchor', return_value=None), \
                patch('module.campaign.chapter_flow.chapter_overlay') as reader:
            self.assertFalse(self.flow.handle(self.field))
        reader.assert_not_called()
        self.session.tap_chapter_overlay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
