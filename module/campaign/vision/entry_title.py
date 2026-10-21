"""固定备战标题的轻量识别；裁切字形后使用单例CPU识别器，不持有游戏输入。"""

from functools import lru_cache
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


MODEL_DIRECTORY = Path(__file__).resolve().parents[3] / 'data/campaign_models/ppocrv6-small'
MODEL_REVISION = 'b8f84f0b80c529de40b4fbb3544b84fa7233a513'
MODEL_FILES = {
    'inference.onnx': '5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634',
    'inference.yml': 'ab078671bb49f06228eadccd34f1bb501e157f7a047095ffb943ba81512c77d1',
}


def title_crop(field, stretch=1):
    """裁固定白色备战栏的深色字形，去掉宽留白；只用于固定UI，不筛选场景道路。

    输入为1776×999 BGR，文字区域必须有足够白底和至少40×18像素字形支持。
    stretch只接受1或2；额外横向拉宽用于旧轻量模型的窄字体回退，不替换或补写字符。
    """
    if field.shape != (999, 1776, 3) or field.dtype != np.uint8 or stretch not in (1, 2):
        raise ValueError('Invalid campaign title frame or normalization')
    crop = field[535:580, 620:970]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if np.mean(gray > 180) < .5:
        return None
    ys, xs = np.nonzero(gray < 90)
    if len(xs) < 40 or np.ptp(xs)+1 < 40 or np.ptp(ys)+1 < 18:
        return None
    crop = crop[max(0, int(ys.min())-3):int(ys.max())+4,
                max(0, int(xs.min())-4):int(xs.max())+5]
    return cv2.resize(crop, None, fx=2*stretch, fy=2, interpolation=cv2.INTER_CUBIC)


class EntryTitleRecognizer:
    """校验官方固定模型后做CPU4线程CTC识别；会话仅在当前任务进程缓存一份。"""

    def __init__(self, directory=MODEL_DIRECTORY):
        """权重、配置、字典和输出维度均须一致；不加载Paddle或新增GPU上下文。

        进程退出时由操作系统回收ORT及线程；不启动子进程、监听器或全局常驻服务。
        """
        import onnxruntime as ort
        import yaml
        directory = Path(directory)
        for name, expected in MODEL_FILES.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Campaign title model checksum mismatch: '+name)
        config = yaml.safe_load((directory/'inference.yml').read_text(encoding='utf8'))
        if (config['Global']['model_name'] != 'PP-OCRv6_small_rec'
                or config['PostProcess']['name'] != 'CTCLabelDecode'):
            raise ValueError('Unknown campaign title model contract')
        self.alphabet = ['blank']+config['PostProcess']['character_dict']+[' ']
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
        self.session = ort.InferenceSession(str(directory/'inference.onnx'), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].shape[1:3] != [3, 48] or inputs[0].type != 'tensor(float)'
                or len(outputs) != 1 or outputs[0].shape[-1] != len(self.alphabet)):
            raise ValueError('Unexpected campaign title ONNX dimensions')
        self.input_name = inputs[0].name

    def recognize(self, crop):
        """按官方BGR/48高/零填充归一化，CTC去重去blank；只返回观测文字和概率均值。"""
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) <= 0:
            raise ValueError('Invalid title crop')
        width = math.ceil(48*crop.shape[1]/crop.shape[0])
        if not 1 <= width <= 2048:
            raise ValueError('Title crop exceeds recognition bounds')
        tensor = np.zeros((1, 3, 48, max(320, width)), np.float32)
        tensor[0, :, :, :width] = cv2.resize(crop, (width, 48)).astype(np.float32).transpose(2, 0, 1)/127.5-1.
        prediction = self.session.run(None, {self.input_name: tensor})[0]
        if (prediction.ndim != 3 or prediction.shape[0] != 1 or prediction.shape[1] < 1
                or prediction.shape[-1] != len(self.alphabet)
                or not np.isfinite(prediction).all() or np.any(prediction < 0) or np.any(prediction > 1)
                or not np.allclose(prediction.sum(axis=-1), 1., atol=.005)):
            raise ValueError('Invalid title CTC probabilities')
        ids = prediction[0].argmax(axis=-1)
        keep = (ids != 0) & np.r_[True, ids[1:] != ids[:-1]]
        text = ''.join(self.alphabet[index] for index in ids[keep])
        score = float(prediction[0].max(axis=-1)[keep].mean()) if keep.any() else 0.
        return dict(rec_text=text, rec_score=score)


@lru_cache(maxsize=1)
def title_recognizer():
    """当前任务最多建立一份识别会话；未安装专用权重时保留原OCR回退，不联网下载。"""
    if not all((MODEL_DIRECTORY/name).exists() for name in MODEL_FILES):
        return None
    return EntryTitleRecognizer()


def read_title(field):
    """只识别有真实标题字形的当前帧，缺模型或缺字形返回None，不生成关卡身份。"""
    crop = title_crop(field)
    if crop is None:
        return None
    reader = title_recognizer()
    return None if reader is None else reader.recognize(crop)
