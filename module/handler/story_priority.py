"""Process the story Skip control before task-specific actions see the frame."""

import time

from module.logger import logger


def skip_story_frame(device):
    from module.ui.assets import SKIP
    now = time.monotonic()
    if now < getattr(device, '_story_skip_after', 0):
        return False
    # Template matching is intentional: a gray region alone is not a Skip button.
    if SKIP.match(device.image, offset=(150, 10), threshold=0.9):
        logger.info('Story priority: skip dialogue')
        device.click(SKIP)
        device._story_skip_after = now + 0.8
        device._story_skip_pending = now + 8
        device.sleep(0.2)
        return True
    # Only inspect confirmations immediately after a recognized story Skip.
    if now < getattr(device, '_story_skip_pending', 0):
        from module.handler.assets import CONFIRM_A, CONFIRM_B, CONFIRM_C
        if not any(button.match(device.image, offset=30, threshold=0.9)
                   for button in (CONFIRM_A, CONFIRM_B, CONFIRM_C)):
            return False
        from module.ocr.ocr import Ocr
        result = Ocr((80, 400, 650, 900), model_type=device.config.Optimization_OcrModelType,
                     name='STORY_SKIP_CONFIRM').ocr(device.image)
        text = result['text'].lower()
        if any(word in text for word in ('跳过', '跳過', 'skip')):
            for item in result['details']:
                if item['text'].strip().lower() in ('确认', '確認', '确定', '確定', 'confirm', 'ok'):
                    box = item['bbox']
                    if len(box) == 4:
                        device.click_xy(80 + int((box[0] + box[2]) / 2),
                                        400 + int((box[1] + box[3]) / 2))
                        device._story_skip_pending = 0
                        return True
    return False
