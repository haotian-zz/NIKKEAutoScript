"""安装固定官方版本的备战标题权重；模型不随源码提交，运行识别时不触发下载。"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

from module.campaign.vision.entry_title import MODEL_DIRECTORY, MODEL_FILES, MODEL_REVISION


def digest(path):
    """分块校验模型文件，避免同时保留整份权重字节和推理会话。"""
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def prepare(source=None):
    """从已有缓存或官方HTTPS地址安装同一模型，全部SHA256正确后原子发布目录。

    已有完整目录只校验，不覆盖；已有损坏或未知目录报错。单文件上限32MiB，网络超时30秒，
    任一步失败由临时目录上下文清理，保留正式模型和其他OCR依赖。
    """
    if MODEL_DIRECTORY.exists():
        if not all((MODEL_DIRECTORY/name).is_file() and digest(MODEL_DIRECTORY/name) == expected
                   for name, expected in MODEL_FILES.items()):
            raise ValueError('Existing campaign title model differs; preserve directory for inspection')
        return MODEL_DIRECTORY
    MODEL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='entry-ocr-', dir=MODEL_DIRECTORY.parent) as temporary:
        candidate = Path(temporary)/'model'
        candidate.mkdir()
        for name, expected in MODEL_FILES.items():
            if source is None:
                url = f'https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/{MODEL_REVISION}/{name}'
                stream = urlopen(url, timeout=30)
            else:
                stream = (source/name).open('rb')
            with stream, (candidate/name).open('wb') as target:
                size = 0
                for block in iter(lambda: stream.read(65536), b''):
                    size += len(block)
                    if size > 32*1024*1024:
                        raise ValueError('Campaign title asset exceeds size limit')
                    target.write(block)
            if digest(candidate/name) != expected:
                raise ValueError('Campaign title asset checksum mismatch: '+name)
        (candidate/'manifest.json').write_text(json.dumps(dict(
            repository='PaddlePaddle/PP-OCRv6_small_rec_onnx', revision=MODEL_REVISION,
            sha256=MODEL_FILES), indent=2)+'\n', encoding='utf8')
        candidate.rename(MODEL_DIRECTORY)
    return MODEL_DIRECTORY


def main():
    """显式准备权重；可指定--source复用已校验缓存，不自动加载模型或启动游戏。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    print(prepare(args.source), flush=True)


if __name__ == '__main__':
    main()
