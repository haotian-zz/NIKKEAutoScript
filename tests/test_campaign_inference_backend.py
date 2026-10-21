"""本机推理回包必须匹配模型、画面和坐标；服务故障只回退一次，不发送游戏输入。"""

import hashlib
import json
from unittest.mock import Mock, patch
import unittest

import numpy as np

from module.campaign.vision.inference_backend import LocalInference


class InferenceBackendTests(unittest.TestCase):
    """隔离服务替身验证生产客户端，不需要GPU或真实网络。"""

    def test_bound_result_is_accepted(self):
        """同图同模型且框在图内，保留坐标；不额外做模型或阈值替换。"""
        image = np.zeros((30, 40, 3), np.uint8)
        item = dict(label='ring', confidence=.9, box=[5., 5., 25., 25.])
        result = dict(schema=1, shape=list(image.shape), model_sha256='model',
                      source_sha256=hashlib.sha256(image.tobytes()).hexdigest(), detections=[item])
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps(result).encode()
        with patch('http.client.HTTPConnection', return_value=connection):
            self.assertEqual(LocalInference('model', ['ring']).predict(image, ['ring'], [True]*4), [item])

    def test_wrong_image_hash_disables_remote_for_the_instance(self):
        """错误原图回包不会使用；后续不重复访问坏服务，交给既有CPU重新计算。"""
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps(dict(schema=1, source_sha256='wrong')).encode()
        backend = LocalInference('model', ['ring'])
        with patch('http.client.HTTPConnection', return_value=connection) as factory:
            for _ in range(2):
                self.assertIsNone(backend.predict(np.zeros((30, 40, 3), np.uint8), ['ring'], [True]*4))
        self.assertEqual(factory.call_count, 1)

    def test_network_failure_falls_back_without_raising(self):
        """服务退出不会阻断已有CPU路径，也不会在每片重复等待两秒。"""
        connection = Mock()
        connection.request.side_effect = OSError('closed')
        backend = LocalInference('model', ['ring'])
        with patch('http.client.HTTPConnection', return_value=connection):
            self.assertIsNone(backend.predict(np.zeros((30, 40, 3), np.uint8), ['ring'], [True]*4))
        self.assertFalse(backend.available)
