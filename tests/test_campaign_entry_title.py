"""独立ONNX识别器的CTC边界保护，不下载模型、不持有游戏窗口或输入。"""

from types import SimpleNamespace
import unittest

import numpy as np

from module.campaign.vision.entry_title import EntryTitleRecognizer, title_crop


class EntryTitleContractTests(unittest.TestCase):
    """验证重复字符经blank分隔后保留，并拒绝空输出和非概率张量。"""

    def test_ctc_repeats_and_blank_preserve_full_identity(self):
        """连续相同字符合并，blank隔开的两个1仍分别保留，编号中的连接符不得补写。"""
        reader = EntryTitleRecognizer.__new__(EntryTitleRecognizer)
        reader.alphabet = ['blank', '2', '1', '-', 'STAGE']
        reader.input_name = 'x'
        ids = [1, 1, 2, 0, 3, 3, 2, 0, 4]
        prediction = np.full((1, len(ids), 5), .0025, np.float32)
        prediction[0, np.arange(len(ids)), ids] = .99
        reader.session = SimpleNamespace(run=lambda *_: [prediction])
        result = reader.recognize(np.zeros((48, 240, 3), np.uint8))
        self.assertEqual(result['rec_text'], '21-1STAGE')
        self.assertAlmostEqual(result['rec_score'], .99, places=5)
        for invalid in [np.zeros((1, 0, 5), np.float32), np.ones((1, 2, 5), np.float32),
                        np.full((1, 2, 5), np.nan, np.float32)]:
            reader.session = SimpleNamespace(run=lambda *_, value=invalid: [value])
            with self.assertRaises(ValueError):
                reader.recognize(np.zeros((48, 240, 3), np.uint8))

    def test_crop_rejects_absent_title_instead_of_recognizing_arbitrary_scene(self):
        """黑色场景和空白标题均不送模型；非法客户区尺寸不得自动缩放成有效画面。"""
        for value in (0, 255):
            self.assertIsNone(title_crop(np.full((999, 1776, 3), value, np.uint8)))
        with self.assertRaises(ValueError):
            title_crop(np.zeros((720, 1280, 3), np.uint8))


if __name__ == '__main__':
    unittest.main()
