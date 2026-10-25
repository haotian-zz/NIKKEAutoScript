"""本机推理回包必须匹配模型、画面和坐标；服务故障只回退一次，不发送游戏输入。"""

import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
import unittest

import numpy as np

from module.campaign.vision import detection
from module.campaign.vision.inference_backend import LocalInference, ensure_worker, stop_worker


class InferenceBackendTests(unittest.TestCase):
    """隔离服务替身验证生产客户端，不需要GPU或真实网络。"""

    def test_owned_worker_shutdown_waits_for_venv_child_and_escalates_when_needed(self):
        """启动器和子解释器分别等待退出，慢子进程可被终止；不扫描未知服务或其他实例。"""
        process, child, parent = Mock(pid=123), Mock(), Mock()
        process.poll.return_value = None
        parent.children.return_value = [child]
        with patch('module.campaign.vision.inference_backend.psutil.Process', return_value=parent) as lookup, \
                patch('module.campaign.vision.inference_backend.psutil.wait_procs',
                      side_effect=[([], [child]), ([child], [])]):
            stop_worker(process)
        lookup.assert_called_once_with(123)
        parent.children.assert_called_once_with(recursive=True)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5.)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()

    def test_unowned_or_exited_worker_is_not_searched_or_terminated(self):
        """复用服务的None句柄或已经结束的本次worker不触发全机进程枚举。"""
        process = Mock();process.poll.return_value = 0
        with patch('module.campaign.vision.inference_backend.psutil.Process') as lookup:
            stop_worker(None)
            stop_worker(process)
        lookup.assert_not_called()
        process.terminate.assert_not_called()

    def test_worker_start_interruption_releases_the_created_process(self):
        """冷启动等待被异常打断时也须清理刚创建的服务，不依赖拿到返回句柄后的finally。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('data/vision_runtime/venv/Scripts/python.exe',
                         'module/campaign/vision/inference_worker.py'):
                path = root/name;path.parent.mkdir(parents=True, exist_ok=True);path.touch()
            process = Mock();process.poll.return_value = None
            connection = Mock();connection.request.side_effect = OSError('not ready')
            with patch('module.campaign.vision.inference_backend.http.client.HTTPConnection', return_value=connection), \
                    patch('module.campaign.vision.inference_backend.subprocess.Popen', return_value=process), \
                    patch('module.campaign.vision.inference_backend.time.sleep', side_effect=RuntimeError('cancelled')), \
                    patch('module.campaign.vision.inference_backend.stop_worker') as stop:
                with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                    ensure_worker(root, 'model')
            stop.assert_called_once_with(process)

    def test_actor_camera_port_keeps_same_frame_and_model_contract(self):
        """人物框原型只改变固定回环端口，仍须匹配原图、权重和有效框，不返回地面点。"""
        field = np.zeros((999, 1776, 3), np.uint8)
        item = dict(label='scene_squad_body', confidence=.95, box=[788., 437., 991., 548.])
        connection = Mock()
        response = connection.getresponse.return_value
        response.status = 200
        response.read.return_value = json.dumps(dict(schema=1, shape=list(field.shape),
            model_sha256='actor', source_sha256=hashlib.sha256(field.tobytes()).hexdigest(),
            detections=[item])).encode()
        with patch('http.client.HTTPConnection', return_value=connection) as factory:
            self.assertEqual(LocalInference('actor', ['scene_squad_body'], port=12279).scene(
                field, ['scene_squad_body']), [item])
        factory.assert_called_once_with('127.0.0.1', 12279, timeout=2.)
        self.assertNotIn('ground', item)

    def test_arbitrary_destination_port_is_rejected(self):
        """不将隔离端口扩展为任意网络转发器，未知端口在创建连接前拒绝。"""
        with self.assertRaises(ValueError):
            LocalInference('actor', ['scene_squad_body'], port=9999)

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


class InferenceConfigurationTests(unittest.TestCase):
    """AutoPush 的 CUDA 优先放进隔离 worker，推图结束关闭；起不来才用进程内 CUDA。"""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        metadata = json.loads((detection.MODEL_DIRECTORY/'campaign.json').read_text(encoding='utf8'))
        self.settings = Path(self.folder.name)/'data/campaign_detector.json'
        self.settings.parent.mkdir(parents=True)
        self.settings.write_text(json.dumps(dict(backend='cuda', threads=4, benchmark_passed=True,
                                                 model_sha256=metadata['sha256'])), encoding='utf8')
        patcher = patch.object(detection, 'DETECTOR_SETTINGS', self.settings)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(setattr, detection, '_worker', None)
        self.addCleanup(setattr, detection, '_detector_profile', {})
        detection._worker, detection._detector_profile = None, {}

    def configure(self, ready, cuda=True, process=None):
        backend = 'module.campaign.vision.inference_backend'
        with patch(f'{backend}.ensure_worker', return_value=process) as start, \
                patch(f'{backend}.worker_ready', return_value=ready), \
                patch(f'{backend}.prepare_cuda', return_value=cuda) as local:
            detection.configure_inference('AutoPush')
        return start, local

    def test_ready_worker_is_used_and_reused_across_segments(self):
        process = Mock()
        process.poll.return_value = None
        start, local = self.configure(ready=True, process=process)
        self.assertEqual(detection._detector_profile, dict(threads=4, remote=True))
        local.assert_not_called()
        start, _ = self.configure(ready=True, process=Mock())
        start.assert_not_called()
        self.assertIs(detection._worker, process)

    def test_worker_unavailable_falls_back_to_in_process_cuda_then_cpu(self):
        self.configure(ready=False)
        self.assertEqual(detection._detector_profile,
                         dict(threads=4, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']))
        self.configure(ready=False, cuda=False)
        self.assertEqual(detection._detector_profile, dict(threads=4))

    def test_other_instances_never_start_a_worker(self):
        with patch('module.campaign.vision.inference_backend.ensure_worker') as start:
            detection.configure_inference('DailyFull')
        start.assert_not_called()
        self.assertEqual(detection._detector_profile, {})

    def test_release_stops_owned_worker_and_drops_sessions(self):
        process = Mock()
        detection._worker, detection._detector_profile = process, dict(threads=4, remote=True)
        with patch('module.campaign.vision.inference_backend.stop_worker') as stop, \
                patch.object(detection.detector, 'cache_clear') as clear:
            detection.release_inference()
        stop.assert_called_once_with(process)
        clear.assert_called_once()
        self.assertIsNone(detection._worker)
        self.assertEqual(detection._detector_profile, {})
