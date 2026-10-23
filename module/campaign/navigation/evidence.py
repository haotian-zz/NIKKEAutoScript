"""移动和实验命令共用的证据写盘检查，不持有窗口或实验全局状态。"""

import json
import time
from datetime import datetime

import cv2
import numpy as np


def write_image(path, image, *args):
    """保存证据图片并检查 OpenCV 返回值，写盘失败必须上报而不能继续假装已留证。

    path 为输出路径，image 为 OpenCV 图像，附加参数直接传给 imwrite。
    写入返回失败时抛出 OSError，避免生成成功回执却缺失现场证据。
    """
    if image is None or not cv2.imwrite(str(path), image, *args):
        raise OSError(f'Could not save evidence image: {path}')
    return True


# 受阻决策关键帧另存一份不回绕的副本：实录 2026-10-21（22-27）目标旁观察帧几分钟后即被32槽覆盖，
# 无法回放“为什么没找到机关”。每会话最多 KEEP_LIMIT 份。
KEEP_EVENTS = frozenset(('goal_view_result', 'switch_search_view', 'switch_search_result', 'structure_attempt',
                         'switch_activated', 'structure_wait_timeout', 'gate_enemy_attempt', 'structure_probe',
                         'facility_survey_view', 'chapter_sweep_motion',
                         # 巡检因离屏头像未确认而中止时要看到当时画面（实录 2026-10-22 22-27）。
                         'offscreen_map_state', 'offscreen_unoccluded', 'offscreen_unoccluded_confirm',
                         'switch_search_unmeasured', 'switch_search_hint',
                         # 任务定位后看到了什么（实录 2026-10-23 24-27：定位后零红圈，32槽很快被巡检覆盖）。
                         'quest_guide', 'goal_view', 'marker_probe', 'quest_barrier_blocked', 'marker_walk'))
KEEP_LIMIT = 300


class ActionEvidence:
    """固定容量记录动作前后帧及原始上下文；每个会话单独归档，恢复流程不会覆盖上一会话。

    后帧使用输入后第一次正式截图，不额外截屏或等待；记录时刻不等同于运动停稳。
    图像只写本机 log，不参与定位判定，也不保存账号配置。
    """

    def __init__(self, root, capacity=32):
        """建立会话目录，每次最多保留 capacity 个事件；槽位回绕时同时清理旧图和元数据。"""
        self.folder = root / 'traces' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.folder.mkdir(parents=True, exist_ok=True)
        self.capacity, self.sequence, self.pending, self.kept = capacity, 0, None, 0

    def record(self, event, field=None, arrays=None, **context):
        """保存一个证据事件并返回槽位；上下文只含动作、章节、分数和坐标，数组按原数值序列化。"""
        slot = self.folder / f'{self.sequence % self.capacity:02d}'
        for path in self.folder.glob(slot.name + '-*.jpg'):
            path.unlink()
        slot.with_suffix('.npz').unlink(missing_ok=True)
        data = dict(sequence=self.sequence, event=event, captured_at=time.time(), **context)
        self.sequence += 1
        if field is not None:
            write_image(str(slot) + '-before.jpg', field, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if arrays:
            np.savez_compressed(slot.with_suffix('.npz'), **arrays)
        self.save(slot, data)
        if event in KEEP_EVENTS and self.kept < KEEP_LIMIT:
            keep = self.folder / 'keep'
            keep.mkdir(exist_ok=True)
            name = keep / f'{data["sequence"]:04d}-{event}'
            if field is not None:
                write_image(str(name) + '.jpg', field, [cv2.IMWRITE_JPEG_QUALITY, 88])
            self.save(name, data)
            self.kept += 1
        return slot, data

    def save(self, slot, data):
        """原子替换单槽 JSON，避免停止期间留下半份元数据。"""
        temporary = slot.with_suffix('.tmp')
        content = json.dumps(data, ensure_ascii=False, indent=2,
                             default=lambda value: value.tolist() if hasattr(value, 'tolist') else str(value))
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(slot.with_suffix('.json'))

    def before(self, event, field, **context):
        """登记即将发送的手势，后续第一张正式截图补全 after；没有后帧时保持证据不完整。"""
        self.pending = self.record(event, field, **context)

    def observed(self, field):
        """只补录最近手势的第一张后帧；无待记录动作时不写盘，避免拖慢普通轮询。"""
        if self.pending is None:
            return
        slot, data = self.pending
        write_image(str(slot) + '-after.jpg', field, [cv2.IMWRITE_JPEG_QUALITY, 88])
        data['after_captured_at'] = time.time()
        self.save(slot, data)
        self.pending = None
