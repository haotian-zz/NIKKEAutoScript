"""局部对应点标定的来源、身份歧义与单次校准候选边界。"""

import copy
import unittest

import cv2
import numpy as np

from module.campaign.navigation.landmark_calibration import (
    calibration_point, local_candidate, stable_map_landmarks, stable_scene_triple,
)


def observed_document():
    """构造独立视野中的小地图敌人，viewport位置故意与实际小队无关。"""
    rows = []
    for view in (2, 3):
        for sample in (0, 1):
            rows.append(dict(view=view, sample=sample, status='accepted',
                             registration=dict(image_sha256='map-a', position_kind='viewport', position=[999., 999.]),
                             enemies=[dict(kind='normal', confidence=.95, map_position=point)
                                      for point in ([608., 267.], [521., 211.], [579., 199.])]))
    return dict(scene_inputs=0, navigation_decision=dict(location=dict(status='accepted', map_sha256='map-a')),
                expanded_correspondences=dict(samples=rows))


class LandmarkCalibrationTests(unittest.TestCase):
    """仅验证数学及输入边界，不加载模型，也不连接游戏或声明四点拟合已可执行。"""

    def candidate(self):
        """使用现场派生坐标构造非共线候选，框架不将拟合成功认作精度通过。"""
        landmarks = stable_map_landmarks(observed_document())
        return local_candidate(landmarks, [[1536., 405.], [1465., 334.], [1375., 344.]],
                               [206., 351.], [891., 503.])

    def test_map_landmarks_require_two_views_and_same_version(self):
        """重复同一视野不算独立地图核验，混合底图立即拒绝。"""
        document = observed_document()
        self.assertEqual(len(stable_map_landmarks(document)['positions']), 3)
        single = copy.deepcopy(document)
        for row in single['expanded_correspondences']['samples']:
            row['view'] = 2
        with self.assertRaises(ValueError):
            stable_map_landmarks(single)
        mixed = copy.deepcopy(document)
        mixed['expanded_correspondences']['samples'][0]['registration']['image_sha256'] = 'map-b'
        with self.assertRaisesRegex(ValueError, '不同底图'):
            stable_map_landmarks(mixed)

    def test_scene_triple_rejects_missing_or_competing_frames(self):
        """少一个脚圈、同帧近邻竞争和大幅移动均不能用历史三点补齐。"""
        values = [dict(kind='enemy_ring', confidence=.9, position=point)
                  for point in ([400., 300.], [500., 300.], [450., 390.])]
        self.assertEqual(stable_scene_triple(values, values).shape, (3, 2))
        with self.assertRaises(ValueError):
            stable_scene_triple(values, values[:2])
        moved = copy.deepcopy(values)
        moved[0]['position'] = [420., 300.]
        with self.assertRaises(ValueError):
            stable_scene_triple(values, moved)

    def test_viewport_is_not_used_as_squad_and_scene_input_source_rejected(self):
        """视野假位置不进入三个敌人坐标，来源中发生过场景输入则不能沿用。"""
        document = observed_document()
        points = stable_map_landmarks(document)['positions']
        self.assertNotIn([999., 999.], points)
        document['scene_inputs'] = 1
        with self.assertRaisesRegex(ValueError, '零场景输入'):
            stable_map_landmarks(document)

    def test_four_point_fit_remains_unvalidated(self):
        """四点精确拟合只有训练残差，没有独立移动点，不能批准路线。"""
        result = self.candidate()
        self.assertEqual(result['status'], 'unvalidated')
        self.assertEqual(result['independent_motion_points'], 0)
        self.assertEqual(result['approved_routes'], 0)
        actual = cv2.perspectiveTransform(np.float32(result['source_points'])[None],
                                          np.asarray(result['matrix']))[0]
        self.assertLess(np.max(np.linalg.norm(actual-np.asarray(result['map_points']), axis=1)), .001)

    def test_symmetric_identity_cannot_be_disambiguated_by_fit(self):
        """等边三角有多个同样好的对应，不能挑一个零误差解伪造身份。"""
        points = [[400., 300.], [500., 300.], [450., 386.60254]]
        with self.assertRaisesRegex(ValueError, '歧义'):
            local_candidate(dict(map_sha256='map-a', positions=points), points, [200., 450.], [200., 450.])

    def test_calibration_step_stays_inside_support_and_cannot_use_unknown_road(self):
        """候选在实际凸包及光幕同侧，未知道路不允许为了实验放宽。"""
        result = self.candidate()
        pillars = [[990., 436.], [1162., 541.]]
        point = calibration_point(result, pillars, np.ones((800, 900), np.uint8))
        hull = np.asarray(result['support'], np.float32).reshape(-1, 1, 2)
        self.assertGreaterEqual(cv2.pointPolygonTest(hull, tuple(point['click']), True), 2.)
        self.assertFalse(point['action_approved'])
        with self.assertRaisesRegex(ValueError, '没有同时满足'):
            calibration_point(result, pillars, np.zeros((800, 900), np.uint8))


if __name__ == '__main__':
    unittest.main()
