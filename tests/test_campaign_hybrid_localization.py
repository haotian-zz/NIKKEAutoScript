"""混合定位的实测圆环、地图版本、输入闸门和设施结果边界。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign.navigation import hybrid_localization as hybrid


def accepted(position=(100., 200.), digest='map-a'):
    """创建已独立确认的实测小队记录，不包含任何道路或传送许可。"""
    return dict(status='accepted', identity=[21, 'normal'], map_sha256=digest,
                position=list(position), measured_ring=[60., 220.])


class HybridLocalizationTests(unittest.TestCase):
    """地图搜索必须找实际圆环，位置变化不能自动升级为设施完成。"""

    def test_pair_rejects_absent_marker_changed_map_and_inconsistent_position(self):
        """高分位置也不能弥补无实测圆环、底图变化或跨帧漂移。"""
        first = accepted()
        self.assertFalse(hybrid.confirmed_pair(first, dict(first, measured_ring=None)))
        self.assertFalse(hybrid.confirmed_pair(first, accepted(digest='map-b')))
        self.assertFalse(hybrid.confirmed_pair(first, accepted((150., 200.))))
        self.assertTrue(hybrid.confirmed_pair(first, accepted((102., 202.))))

    def test_displacement_remains_unconfirmed_transfer_or_activation(self):
        """大位移、微小位移和跨图结果均不得产生已传送或已开门。"""
        for after, expected in ((accepted((1000., 200.)), 'moved'),
                                (accepted((101., 200.)), 'position_unchanged'),
                                (accepted(digest='map-b'), 'unknown')):
            result = hybrid.position_feedback(accepted(), after)
            self.assertEqual(result['state'], expected)
            self.assertEqual(result['transfer'], 'unconfirmed')
            self.assertEqual(result['activated'], 'unconfirmed')

    def session(self, snapshots):
        """构造可检查正式UI闸门的会话，不连接游戏或加载模型。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        owner = Mock()
        owner.observer.matcher.digest = 'map-a'
        owner.observer.observe.side_effect = snapshots
        owner.expand_minimap.return_value = owner.collapse_minimap.return_value = True
        owner.minimap_expanded.return_value = True
        session = Mock(owner=owner, expected_identity=(21, 'normal'), ui_validator=None)
        session.win.client.return_value = field
        session.win.client_origin.return_value = (10, 20)
        return session

    def snapshot(self, good=False, mode='expanded'):
        """区分实测小队与无标记视野，缺失时不填补位置。"""
        return dict(localization='accepted' if good else 'needs_review',
                    squad=[100., 200.] if good else None, road_iou=.9 if good else None,
                    minimap=dict(mode=mode, squad=[60., 220.] if good else None))

    def test_searches_new_panel_then_requires_two_measurements_and_restores_ui(self):
        """初始视野无圆环时搜索相邻面板，双帧通过后收起，零场景点击。"""
        session = self.session([self.snapshot(mode='compact'), self.snapshot(),
                                self.snapshot(True), self.snapshot(True)])
        result = hybrid.locate_squad(session)
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['position'], [100., 200.])
        self.assertEqual(len(result['observations']), 4)
        session.win.handler.mouse_swipe.assert_called_once()
        session.tap.assert_not_called()
        session.owner.collapse_minimap.assert_called_once()
        self.assertIsNone(session.ui_validator)
        self.assertEqual(result['approved_routes'], 0)

    def test_missing_panel_stops_before_swipe_and_still_attempts_safe_collapse(self):
        """面板中途消失时禁止把平移发到场景，保留清理闸门。"""
        session = self.session([self.snapshot(mode='compact'), self.snapshot()])
        session.owner.minimap_expanded.return_value = False
        with self.assertRaisesRegex(RuntimeError, '面板消失'):
            hybrid.locate_squad(session)
        session.win.handler.mouse_swipe.assert_not_called()
        session.owner.collapse_minimap.assert_called_once()


if __name__ == '__main__':
    unittest.main()
