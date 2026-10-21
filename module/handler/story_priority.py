"""在任务读取画面前处理剧情跳过，兼容日常竖屏与主线横屏。"""

import time
import math

import cv2

from module.logger import logger


def portrait_story_menu(image, reader):
    """以四个完整菜单词复核720×1280剧情页，补足发光SKIP的低模板分数。

    只在低分模板已提出右上角按钮时调用；四行固定字区均须匹配HIDE/AUTO/LOG/SKIP
    且分数至少0.95。没有用文字识别结果直接生成点击坐标，按钮仍来自当帧模板提案。
    """
    if image.shape != (1280, 720, 3):
        return False
    areas = ((286, 19, 347, 45), (396, 19, 463, 45), (510, 19, 560, 45), (614, 19, 675, 45))
    crops = [cv2.resize(image[top:bottom, left:right, ::-1], None, fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC) for left, top, right, bottom in areas]
    results = reader.paddleocr.recognize(crops)
    if len(results) != 4:
        return False
    for item, expected in zip(results, ('HIDE', 'AUTO', 'LOG', 'SKIP')):
        score = float(item.get('rec_score', 0.))
        if item.get('rec_text', '').strip().upper() != expected or not math.isfinite(score) or score < .95:
            return False
    return True


def skip_story_frame(device):
    """优先跳过本帧剧情及其确认框；返回 True 表示需要重新截图再交给任务。

    横屏只在 1776×999 右上角匹配已校准的 SKIP 模板，竖屏保持原匹配范围。
    防抖期间仍返回 True，避免把尚未关闭的剧情帧交给移动逻辑。确认框只在刚点击
    已识别的 SKIP 后八秒内检查，文字必须同时包含跳过说明和精确确认按钮。
    """
    from module.ui.assets import SKIP
    now = time.monotonic()
    wide = device.image.shape[:2] == (999, 1776)
    if now < getattr(device, '_story_skip_pending', 0):
        possible = wide
        if not wide:
            from module.handler.assets import CONFIRM_A, CONFIRM_B, CONFIRM_C
            possible = any(button.match(device.image, offset=30, threshold=0.9)
                           for button in (CONFIRM_A, CONFIRM_B, CONFIRM_C))
        if possible:
            from module.ocr.ocr import Ocr
            area = (528, 250, 1248, 800) if wide else (80, 400, 650, 900)
            result = Ocr(area, model_type=device.config.Optimization_OcrModelType,
                         name='STORY_SKIP_CONFIRM').ocr(device.image)
            if any(word in result['text'].lower() for word in ('跳过', '跳過', 'skip')):
                for item in result['details']:
                    if item['text'].strip().lower() in ('确认', '確認', '确定', '確定', 'confirm', 'ok'):
                        box = item['bbox']
                        if len(box) == 4 and now >= getattr(device, '_story_skip_after', 0):
                            device.click_xy(area[0] + int((box[0] + box[2]) / 2),
                                            area[1] + int((box[1] + box[3]) / 2))
                            device._story_skip_pending = 0
                            device._story_skip_after = now + 0.8
                            logger.info('Story priority: confirmed dialogue skip')
                            return True
                return True
    if wide:
        from module.campaign.vision.battle import story_skip
        _, point = story_skip(cv2.cvtColor(device.image, cv2.COLOR_RGB2BGR))
        visible = point is not None
    else:
        visible = SKIP.match(device.image, offset=(150, 10), threshold=0.9)
        point = None
        if not visible and device.image.shape == (1280, 720, 3):
            proposal = SKIP.match(device.image, offset=(150, 10), threshold=.7)
            if proposal:
                left, top, right, bottom = SKIP.button
                candidate = ((left+right)/2., (top+bottom)/2.)
                if 600 <= candidate[0] <= 700 and 10 <= candidate[1] <= 60:
                    from module.ocr.ocr import Ocr
                    reader = Ocr([], model_type=device.config.Optimization_OcrModelType,
                                 name='PORTRAIT_STORY_MENU')
                    if portrait_story_menu(device.image, reader):
                        previous = getattr(device, '_story_portrait_candidate', None)
                        device._story_portrait_candidate = (now, *candidate)
                        if (previous is None or not 0 <= now-previous[0] <= 3.
                                or (candidate[0]-previous[1])**2+(candidate[1]-previous[2])**2 > 25.):
                            return True
                        visible = True
            if not visible:
                device._story_portrait_candidate = None
    if visible:
        if now >= getattr(device, '_story_skip_after', 0):
            logger.info('Story priority: skip dialogue')
            if wide:
                device.click_xy(*[int(round(value)) for value in point])
            else:
                device.click(SKIP)
            device._story_skip_after = now + 0.8
            device._story_skip_pending = now + 8
        return True
    return False
