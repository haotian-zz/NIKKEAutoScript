"""设施底图坐标的换算与闭环寻位；定位和拖动都是替身，不连接游戏。"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

with patch.dict('sys.modules', {'module.logger': SimpleNamespace(logger=Mock())}):
    from module.campaign.navigation import facility_map as facility
from module.campaign.navigation import controls as goto


def report(offset):
    """展开 ROI 到底图只差平移的配准结果：底图 = ROI + offset。"""
    matrix = np.eye(3)
    matrix[:2, 2] = offset
    return dict(roi_to_map=matrix.tolist(), position_kind='viewport')


class FacilityMapTests(unittest.TestCase):
    """坐标只决定往哪看；镜头拖不动、配准失败或小队未确认时停止，不发送走位。"""

    def setUp(self):
        """替身运行器：定位器使用标定的 ROI→参考平面单应。"""
        rng = np.random.default_rng(5)
        self.field = cv2.cvtColor(cv2.resize(rng.integers(0, 256, (250, 444), dtype=np.uint8), (1776, 999)),
                                  cv2.COLOR_GRAY2BGR)
        self.localizer = SimpleNamespace(old_matrix=goto.ROI_TO_PLANE)
        self.runner = SimpleNamespace(session=Mock(localizer=self.localizer),
                                      map_frame=Mock(return_value=self.field))

    def test_gaze_chart_round_trip(self):
        """镜头注视点落在 VIEWPORT_PIXEL；客户区与底图互相换算可逆，ROI 外的点不投影。"""
        chart = facility.gaze_chart(self.localizer)
        view = report([1000., 2000.])
        np.testing.assert_allclose(facility.map_to_client(chart, view, facility.gaze_position(view)),
                                   goto.VIEWPORT_PIXEL, atol=1e-6)
        point = facility.client_to_map(chart, view, [1200., 700.])
        np.testing.assert_allclose(facility.map_to_client(chart, view, point), [1200., 700.], atol=1e-6)
        self.assertIsNone(facility.map_to_client(chart, view, facility.gaze_position(view)+[900., 0.]))

    def test_seek_pans_until_target_projects_into_view(self):
        """目标起初远在视野外：拖一次后重新配准，投影进入画面即停，返回预计落点。"""
        target = np.array([5000., 3000.])
        reports = [report(target-[243.+700., 231.]), report(target-[243., 231.])]
        with patch.object(facility, 'locate_view', side_effect=reports), \
                patch.object(facility, 'squad_arrow', return_value=np.array([888., 600.])), \
                patch.object(facility, 'pan_scene') as pan:
            reached = facility.seek_map_point(self.runner, target)
        self.assertEqual(pan.call_count, 1)
        self.assertTrue(pan.call_args.kwargs['paced'])
        np.testing.assert_allclose(reached['client'], goto.VIEWPORT_PIXEL, atol=1e-6)

    def test_seek_stops_when_camera_does_not_approach_or_squad_unknown(self):
        """拖后注视点没有靠近（地图边界）就停；没有小队箭头也没有离屏标记时一次都不拖。"""
        target = np.array([5000., 3000.])
        far = report(target-[243.+700., 231.])
        with patch.object(facility, 'locate_view', side_effect=[far, far]), \
                patch.object(facility, 'squad_arrow', return_value=np.array([888., 600.])), \
                patch.object(facility, 'pan_scene') as pan:
            self.assertIsNone(facility.seek_map_point(self.runner, target))
        self.assertEqual(pan.call_count, 1)
        with patch.object(facility, 'locate_view', return_value=far), \
                patch.object(facility, 'squad_arrow', return_value=None), \
                patch.object(facility, 'squad_offscreen', return_value=None), \
                patch.object(facility, 'pan_scene') as pan:
            self.assertIsNone(facility.seek_map_point(self.runner, target))
        pan.assert_not_called()

    def test_record_positions_only_when_missing_and_reports_scene_change(self):
        """已有坐标不再开地图；缺坐标时配准一次，按注视点换算并报告开合地图后画面是否未变。"""
        items = [dict(facility_id='scene-0001', position=list(goto.VIEWPORT_PIXEL))]
        catalog = Mock()
        catalog.needs_map.return_value = []
        with patch.object(facility, 'locate_view') as locate:
            self.assertIsNone(facility.record_map_positions(self.runner, catalog, items, self.field))
        locate.assert_not_called()
        catalog.needs_map.return_value = items
        catalog.place.return_value = []
        with patch.object(facility, 'locate_view', return_value=report([100., 200.])):
            moved = facility.record_map_positions(self.runner, catalog, items, self.field)
        self.assertTrue(moved['unchanged'])
        convert = catalog.place.call_args.args[1]
        np.testing.assert_allclose(convert(goto.VIEWPORT_PIXEL), np.add(goto.EXPANDED_CENTER, [100., 200.]),
                                   atol=1e-6)
        with patch.object(facility, 'locate_view', side_effect=ValueError('未通过')):
            self.assertIsNone(facility.record_map_positions(self.runner, catalog, items, self.field))


if __name__ == '__main__':
    unittest.main()
