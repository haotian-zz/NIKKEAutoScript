"""遮挡补查只允许已核验地图控件；双帧头像失败和异常均恢复面板。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from module.campaign.vision import minimap_occlusion as occlusion
from module.campaign.vision.minimap import minimap_closed_control_visible


class MinimapOcclusionTests(unittest.TestCase):
    """替身只记录UI调用，不连接游戏；头像位置不能由控件或画面中心补齐。"""

    def test_x_without_verified_minimap_cannot_be_clicked(self):
        """场景中的X或没有地图控件的画面，均不能变成收起按钮。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        cv2.line(field, (197, 94), (211, 108), (255, 255, 255), 3)
        cv2.line(field, (197, 108), (211, 94), (255, 255, 255), 3)
        with patch.object(occlusion, 'map_visible', return_value=False):
            self.assertIsNone(occlusion.compact_close_point(field))
        with patch.object(occlusion, 'map_visible', return_value=True), \
                patch.object(occlusion, 'minimap_control_visible', return_value=True):
            point = occlusion.compact_close_point(field)
            self.assertLess(np.linalg.norm(np.asarray(point)-[204, 101]), 3.)

    def test_real_closed_pin_is_distinct_from_magnifier(self):
        """实机图钉二值轮廓的小孔不能再按放大镜80px孔洞规则漏掉；圆环及亮块不匹配。"""
        packed = bytes.fromhex('03fc01ffc07ffc1fffc7fffcffff9f83fbf03ffc07ff80fdf81fbf87f7fffcffff8fffe1fffc1fff03ffe'
                               '03ff803ff007fc007f0007e00078000e0000800')
        glyph = np.unpackbits(np.frombuffer(packed, np.uint8))[:26*19].reshape(26, 19)
        control = np.full((44, 44, 3), 70, np.uint8)
        control[9:35, 12:31][glyph > 0] = 255
        self.assertTrue(minimap_closed_control_visible(control))
        field = np.zeros((999, 1776, 3), np.uint8)
        field[76:120, 20:64] = control
        with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
            from module.campaign.vision.observation import CampaignObserver
        self.assertTrue(CampaignObserver.minimap_closed(field))
        blue = cv2.cvtColor(np.uint8([[[100, 230, 210]]]), cv2.COLOR_HSV2BGR)[0, 0]
        field[118:273, 27:207] = blue
        self.assertTrue(CampaignObserver.minimap_closed(field))
        ring = np.zeros_like(control)
        cv2.circle(ring, (22, 22), 12, (255, 255, 255), 4)
        self.assertFalse(minimap_closed_control_visible(ring))
        self.assertFalse(minimap_closed_control_visible(np.full_like(control, 255)))

    def test_two_actual_avatars_required_and_map_restored(self):
        """仅一个头像观测不放行；两次真实同位置头像才返回，两个分支都恢复地图。"""
        marker = dict(center=[85., 315.], radius=32.5)
        for values, expected in (([marker, None], None), ([marker, marker], marker)):
            session = Mock()
            session.win.client.return_value = np.zeros((999, 1776, 3), np.uint8)
            session.owner.observer.minimap_closed.side_effect = [False, True, True, True]
            with patch.object(occlusion, 'compact_close_point', return_value=[204., 101.]), \
                    patch.object(occlusion, 'restore_compact_map', return_value=True) as restore, \
                    patch.object(occlusion, 'squad_offscreen', side_effect=values):
                self.assertEqual(occlusion.unoccluded_offscreen(session), expected)
            session.tap_verified_ui.assert_called_once()
            restore.assert_called_once_with(session)

    def test_read_failure_still_restores_and_never_clicks_avatar(self):
        """隐藏后读取失败也走finally恢复，不发送头像点击或场景拖动。"""
        session = Mock()
        session.win.client.return_value = np.zeros((999, 1776, 3), np.uint8)
        session.owner.observer.minimap_closed.side_effect = [False, True, True]
        with patch.object(occlusion, 'compact_close_point', return_value=[204., 101.]), \
                patch.object(occlusion, 'restore_compact_map', return_value=True) as restore, \
                patch.object(occlusion, 'squad_offscreen', side_effect=ValueError('unreadable')):
            with self.assertRaises(ValueError):
                occlusion.unoccluded_offscreen(session)
        self.assertEqual(session.tap_verified_ui.call_args.args[0], [204., 101.])
        restore.assert_called_once_with(session)
        session.win.handler.mouse_swipe.assert_not_called()

    def test_initially_closed_map_reads_and_restores_without_clicking_missing_x(self):
        """进入时地图已关闭也要复核头像并恢复，不能将缺少X直接返回当作补查成功。"""
        session = Mock()
        session.win.client.return_value = np.zeros((999, 1776, 3), np.uint8)
        session.owner.observer.minimap_closed.return_value = True
        marker = dict(center=[85., 315.], radius=32.5)
        with patch.object(occlusion, 'restore_compact_map', return_value=True) as restore, \
                    patch.object(occlusion, 'squad_offscreen', return_value=marker):
            self.assertEqual(occlusion.unoccluded_offscreen(session), marker)
        session.tap_verified_ui.assert_not_called()
        restore.assert_called_once_with(session)

    def test_recenter_requires_actual_avatar_again_at_final_input(self):
        """显式回中只点真实头像；最终帧头像消失则拒绝，仍恢复地图。"""
        marker = dict(center=[85., 315.], radius=32.5)
        for final_marker, expected_calls in ((marker, 1), (None, 0)):
            session = Mock()
            field = np.zeros((999, 1776, 3), np.uint8)
            session.win.client.return_value = field
            session.owner.observer.minimap_closed.return_value = True
            sent = []

            def send(point, validator):
                """替身仅模拟正式输入前的最后一次视觉核验，不伪造核验通过。"""
                if not validator(field):
                    raise RuntimeError('avatar changed')
                sent.append(point)

            session.tap_verified_ui.side_effect = send
            with patch.object(occlusion, 'restore_compact_map', return_value=True) as restore, \
                    patch.object(occlusion, 'squad_offscreen', side_effect=[marker, marker, final_marker]):
                if final_marker is None:
                    with self.assertRaisesRegex(RuntimeError, 'avatar changed'):
                        occlusion.recenter_offscreen(session)
                else:
                    self.assertEqual(occlusion.recenter_offscreen(session), marker)
            self.assertEqual(len(sent), expected_calls)
            if sent:
                self.assertEqual(sent[0], marker['center'])
            restore.assert_called_once_with(session)
            session.win.handler.mouse_swipe.assert_not_called()

    def test_recenter_missing_second_avatar_sends_no_avatar_click(self):
        """初始地图已关闭但双帧头像不成立时，不点击历史位置或画面中心。"""
        session = Mock()
        session.win.client.return_value = np.zeros((999, 1776, 3), np.uint8)
        session.owner.observer.minimap_closed.return_value = True
        with patch.object(occlusion, 'restore_compact_map', return_value=True) as restore, \
                    patch.object(occlusion, 'squad_offscreen', side_effect=[dict(center=[85., 315.]), None]):
            self.assertIsNone(occlusion.recenter_offscreen(session))
        session.tap_verified_ui.assert_not_called()
        restore.assert_called_once_with(session)


    def test_blue_scene_with_compact_minimap_is_not_expanded(self):
        """实录22章：中部是青蓝地面且紧凑小地图仍在，不能判为展开面板；紧凑面板消失后才看中部。"""
        import ast
        from pathlib import Path
        tree = ast.parse((Path(__file__).resolve().parents[1]/'module/campaign/campaign.py').read_text(encoding='utf-8'))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'minimap_expanded')
        compact = {'visible': True}
        field = np.zeros((999, 1776, 3), np.uint8)
        field[280:742, 644:1130] = 1
        namespace = dict(cv2=cv2, EXPANDED_ROI=(644, 280, 1130, 742),
                         map_visible=lambda crop: bool(crop.any()) or (crop.shape[:2] == (155, 180)
                                                                        and compact['visible']),
                         minimap_control_visible=lambda crop: compact['visible'])
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'campaign', 'exec'), namespace)
        owner = SimpleNamespace()
        self.assertFalse(namespace['minimap_expanded'](owner, field))
        compact['visible'] = False
        self.assertTrue(namespace['minimap_expanded'](owner, field))
        field[280:742, 644:1130] = 0
        self.assertFalse(namespace['minimap_expanded'](owner, field))

if __name__ == '__main__':
    unittest.main()
