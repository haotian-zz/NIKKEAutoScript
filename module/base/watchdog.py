"""Bounded waits with evidence, independent of how often a button is clicked."""

import re
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from module.exception import GameStuckError
from module.logger import logger


def save_wait_image(device, label):
    image = getattr(device, 'image', None)
    if image is None:
        return None
    folder = Path('log/timeout')
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', label)[:70]
    path = folder / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{name}.png'
    Image.fromarray(image).save(path)
    logger.warning(f'Wait screenshot: {path}')
    return path


class WaitGuard:
    def __init__(self, device, label, timeout=90, warn_after=30):
        self.device = device
        self.label = label
        self.timeout = timeout
        self.warn_after = warn_after
        self.progress()

    def progress(self):
        self.started = time.monotonic()
        self.warned = False

    def check(self):
        elapsed = time.monotonic() - self.started
        if elapsed >= self.warn_after and not self.warned:
            self.warned = True
            logger.warning(f'{self.label}: waiting {int(elapsed)}s; saving current screen')
            save_wait_image(self.device, self.label)
        if elapsed >= self.timeout:
            from module.execution.context import report
            save_wait_image(self.device, self.label + '_timeout')
            report('failed', f'{self.label}超时', f'{int(elapsed)} 秒未确认进展，已保存截图', retryable=True)
            raise GameStuckError(f'{self.label}: no confirmed progress within {self.timeout}s')
