"""同机隔离推理客户端；图像只发往固定回环端口，失败回到原CPU推理。"""

import hashlib
import http.client
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import psutil


_cuda_handles = []


def prepare_cuda(repository):
    """给当前AutoPush进程加载经过benchmark验证的私有ORT，不修改生产依赖。

    路径固定在data/vision_runtime，保留CUDA DLL目录句柄直到进程退出；已载入其他ORT时
    不尝试热替换二进制模块。环境缺失或CUDA不可用返回False，由调用者保留CPU路径。
    """
    runtime = Path(repository)/'data/vision_runtime'
    package = runtime/'ort39-gpu'
    libraries = runtime/'venv/Lib/site-packages/torch/lib'
    if not hasattr(os, 'add_dll_directory') or not (package/'onnxruntime/__init__.py').exists():
        return False
    if not libraries.is_dir():
        return False
    existing = sys.modules.get('onnxruntime')
    if existing is not None and package.resolve() not in Path(existing.__file__).resolve().parents:
        return False
    try:
        if not _cuda_handles:
            _cuda_handles.append(os.add_dll_directory(str(libraries)))
            os.environ['PATH'] = str(libraries)+os.pathsep+os.environ.get('PATH', '')
        if str(package) not in sys.path:
            sys.path.insert(0, str(package))
        import onnxruntime as ort
        return 'CUDAExecutionProvider' in ort.get_available_providers()
    except (ImportError, OSError):
        return False


class LocalInference:
    """验证模型哈希、输入哈希及坐标契约，不把服务回包直接当作输入指令。"""

    def __init__(self, model_sha256, labels, port=12280):
        """服务首次不可用后禁用本次实例调用；只允许既有检测12280或隔离人物原型12279端口。"""
        if port not in (12279, 12280):
            raise ValueError('Unsupported local inference port')
        self.model_sha256, self.labels = model_sha256, labels
        self.port = port
        self.available = True

    def predict(self, image, labels, guards):
        """发送无压缩BGR并核对每个框；只保留请求类别，失败返回None供原CPU重算。"""
        return self.request('/detect', image, labels, guards)

    def scene(self, image, labels, legacy=False):
        """整帧只传输一次，在同机服务内完成八片或六片推理，避免逐片HTTP调用。"""
        if image.shape != (999, 1776, 3):
            return None
        return self.request('/legacy' if legacy else '/grid', image, labels, [False]*4)

    def request(self, route, image, labels, guards):
        """所有回包共享模型、原图与坐标验证；服务失败时本次实例停止远端调用。"""
        if not self.available:
            return None
        limit = 1776*999*3 if route != '/detect' else 640*640*3
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not 0 < image.size <= limit:
            return None
        body = np.ascontiguousarray(image).tobytes()
        digest = hashlib.sha256(body).hexdigest()
        metadata = dict(shape=list(image.shape), labels=list(labels), guards=list(guards),
                        source_sha256=digest, model_sha256=self.model_sha256)
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=2.)
        try:
            connection.request('POST', route, body=body,
                               headers={'X-NKAS-Frame': json.dumps(metadata),
                                        'Content-Type': 'application/octet-stream'})
            response = connection.getresponse()
            payload = response.read(262145)
            if response.status != 200 or len(payload) > 262144:
                raise ValueError('Invalid detector response')
            result = json.loads(payload)
            if (result.get('schema') != 1 or result.get('source_sha256') != digest
                    or result.get('model_sha256') != self.model_sha256 or result.get('shape') != list(image.shape)):
                raise ValueError('Detector result belongs to another model or image')
            detections = result['detections']
            if not isinstance(detections, list) or len(detections) > 512:
                raise ValueError('Invalid detection count')
            for item in detections:
                box = np.asarray(item['box'], float)
                score = float(item['confidence'])
                if (item['label'] not in labels or item['label'] not in self.labels or box.shape != (4,)
                        or not np.isfinite(box).all() or not 0 <= score <= 1
                        or not (0 <= box[0] < box[2] <= image.shape[1] and 0 <= box[1] < box[3] <= image.shape[0])):
                    raise ValueError('Invalid detection contract')
            return detections
        except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
            self.available = False
            logging.getLogger('nkas').warning(f'推图检测 worker 不可用（{exc}），本次任务改用进程内 CPU 检测。')
            return None
        finally:
            connection.close()


def worker_ready(model_sha256):
    """健康检查同时核对模型和实际CUDA后端，防止同端口的其他进程被误用。"""
    connection = http.client.HTTPConnection('127.0.0.1', 12280, timeout=.5)
    try:
        connection.request('GET', '/health')
        response = connection.getresponse()
        value = json.loads(response.read(8192))
        return (response.status == 200 and value.get('model_sha256') == model_sha256
                and value.get('providers', [])[:1] == ['CUDAExecutionProvider'])
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


# 冷启动要导入 torch 并建 CUDA 会话：实测 2026-10-25（win4090）热缓存 7.7 秒、冷启动超过 12 秒。
WORKER_START_SECONDS = 30.


def ensure_worker(repository, model_sha256):
    """按本机已验证配置启动隔离worker；不改生产依赖，不接管未知服务。

    返回本次创建的进程句柄，调用者任务结束时关闭；已有同模型服务只复用，不由本任务终止。
    环境缺失或30秒内未准备好则返回None，既有CPU回退仍可运行。
    """
    repository = Path(repository)
    runtime = repository/'data/vision_runtime'

    if worker_ready(model_sha256):
        return None
    python = runtime/'venv/Scripts/python.exe'
    worker = repository/'module/campaign/vision/inference_worker.py'
    if not python.exists() or not worker.exists():
        return None
    with (runtime/'detector-worker.log').open('ab') as log:
        process = subprocess.Popen([str(python), '-u', str(worker), '--code',
                                    str(repository/'module/campaign/vision/detection.py'), '--models',
                                    str(repository/'module/campaign/vision/models')], cwd=str(repository),
                                   stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    started = False
    try:
        deadline = time.monotonic()+WORKER_START_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            if worker_ready(model_sha256):
                started = True
                return process
            time.sleep(.2)
        return None
    finally:
        if not started:
            stop_worker(process)


def stop_worker(process):
    """关闭本任务创建的worker及其已观察子进程，并有界等待资源释放。

    Windows虚拟环境启动器可能另外启动解释器，须在终止父进程前取得子进程身份；
    仅处理该Popen仍存活时的后代，不按进程名、端口或其他实例枚举终止对象。
    """
    if process is None or process.poll() is not None:
        return
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    process.terminate()
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5.)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.)
    _, alive = psutil.wait_procs(children, timeout=5.)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(alive, timeout=5.)
    if remaining:
        raise subprocess.TimeoutExpired('owned inference worker children', 10.)
