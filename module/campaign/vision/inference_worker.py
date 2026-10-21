#!/usr/bin/env python3
"""隔离CUDA推理服务：固定回环端口，只接收BGR图像，不持有游戏输入。"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def load_detection(path):
    """从明确的研究/源码文件加载同一前后处理，避免实验环境导入整个NKAS。"""
    spec = importlib.util.spec_from_file_location('nkas_benchmark_detection', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    """启动前验证CUDA后端确实被使用；不把静默CPU回退记作GPU成功。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', required=True)
    parser.add_argument('--models', required=True)
    args = parser.parse_args()
    import torch
    import onnxruntime as ort
    ort.preload_dlls()
    module = load_detection(args.code)
    detector = module.Detector(args.models, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    if detector.session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA provider failed; refusing a mislabeled benchmark')
    detector.predict(np.zeros((640, 640, 3), np.uint8))
    module.detector = lambda: detector

    class Handler(BaseHTTPRequestHandler):
        """结构化小型回包，原始画面仅存内存，请求不支持任意地址或文件操作。"""

        def do_GET(self):
            """返回真实执行后端及模型身份，用于本机实验准备检查。"""
            self.respond(200, dict(ready=True, model_sha256=detector.metadata['sha256'],
                                   providers=detector.session.get_providers(), onnxruntime=ort.__version__))

        def do_POST(self):
            """核对尺寸、模型及内容哈希后推理，回包坐标只属于本次图像。"""
            try:
                length = int(self.headers.get('Content-Length', '0'))
                limit = 640*640*3 if self.path == '/detect' else 1776*999*3
                if self.path not in ('/detect', '/grid', '/legacy') or not 0 < length <= limit:
                    raise ValueError('Invalid request size')
                metadata = json.loads(self.headers['X-NKAS-Frame'])
                body = self.rfile.read(length)
                shape = metadata['shape']
                if (len(shape) != 3 or shape[2] != 3 or min(shape) <= 0 or np.prod(shape) != len(body)
                        or metadata['model_sha256'] != detector.metadata['sha256']
                        or metadata['source_sha256'] != hashlib.sha256(body).hexdigest()):
                    raise ValueError('Image or model identity mismatch')
                labels, guards = metadata['labels'], metadata['guards']
                if len(guards) != 4 or any(type(value) is not bool for value in guards):
                    raise ValueError('Invalid tile edges')
                image = np.frombuffer(body, np.uint8).reshape(shape)
                if self.path == '/detect':
                    found = detector.predict(image, labels, guards)
                else:
                    if tuple(shape) != (999, 1776, 3):
                        raise ValueError('Unexpected scene geometry')
                    found = module.detect_grid(image, labels) if self.path == '/grid' else module.detect_scene(image, labels)
                self.respond(200, dict(schema=1, shape=shape, model_sha256=detector.metadata['sha256'],
                                       source_sha256=metadata['source_sha256'],
                                       detections=[dict(label=item.label, confidence=item.confidence,
                                                        box=list(item.box)) for item in found]))
            except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
                self.respond(422, dict(error=type(exc).__name__))

        def respond(self, code, value):
            """限制返回字段，不回显图像、请求体或私人文件路径。"""
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            """避免输出截图请求及轮询噪音。"""
            return

    print(json.dumps(dict(ready=True, providers=detector.session.get_providers(), port=12280)), flush=True)
    HTTPServer(('127.0.0.1', 12280), Handler).serve_forever()


if __name__ == '__main__':
    main()
