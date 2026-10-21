"""不同CPU后端的FP32舍入可以容忍，真正越界的模型分数必须继续拒绝。"""

import unittest

import numpy as np

from module.campaign.vision.detection import decode


class ProbabilityRoundingTests(unittest.TestCase):
    """通过实际解码与抑制路径检查，不替换概率或目标筛选逻辑。"""

    def output(self, first, second):
        """生成一框两类的最小原始张量，框位于真实图内且不会触发边缘拒绝。"""
        return np.array([50., 50., 20., 20., first, second], np.float32).reshape(1, 6, 1)

    def test_float32_rounding_keeps_valid_object(self):
        """ARM上实测约1个精度单位负零不能让整帧失败，置信度仍裁回概率范围。"""
        for scores, expected in (((-1.2e-7, .9), 'b'), ((1.0000001, 0.), 'a')):
            found = decode(self.output(*scores), (1., 1.), (0., 0.), dict(a=.5, b=.5), (100, 100), ('a', 'b'))
            self.assertEqual([item.label for item in found], [expected])
            self.assertTrue(0 <= found[0].confidence <= 1)

    def test_materially_invalid_scores_still_fail(self):
        """明显负数或超过1的分数仍视为模型契约错误，不能靠数值兼容放行。"""
        for scores in ((-.01, .9), (1.01, 0.)):
            with self.assertRaises(ValueError):
                decode(self.output(*scores), (1., 1.), (0., 0.), dict(a=.5, b=.5), (100, 100), ('a', 'b'))
