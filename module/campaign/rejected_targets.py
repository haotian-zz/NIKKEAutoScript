"""点过却走不到或点了没反应的脚圈位置；按场景特征配准跟到新画面，换了扫描编号也不再点同一处。

脚圈编号只在相邻两次扫描能跟踪时延续，镜头拖动或观察簿重置后同一处会换新编号、重新被选中。
实录 2026-10-24（25章）：岩浆坑被检测成脚圈，小队走到坑边停住，下一轮换了编号又点过去；
小队站在橙色圈上且头顶没箭头时把自己脚下的圈当敌人，点了 20 秒没任何反应后整轮中止。
这里不看外观判断是不是敌人，只记点击后的实际结果；配准不上的旧记录不排除任何目标。
"""

import cv2
import numpy as np

from module.campaign.facility_memory import same_screen
from module.campaign.vision.enemy_ring import HUD_RECTS, scene_registration
from module.campaign.vision.surface_motion import project

# 新目标离记下的位置不超过这么远（客户区像素）就跳过；脚圈框宽约 70～110。
REJECT_RADIUS = 60.
# 最多保留的记录数；每条在选目标时都要和当前帧配准一次。
MAX_RECORDS = 8


class SceneView:
    """一帧客户区的 SIFT 特征（排除固定 HUD），供 scene_registration 配准。"""

    def __init__(self, field):
        mask = np.full(field.shape[:2], 255, np.uint8)
        for left, top, right, bottom in HUD_RECTS:
            mask[top:bottom, left:right] = 0
        keys, self.descriptors = cv2.SIFT_create(nfeatures=3000).detectAndCompute(
            cv2.cvtColor(field, cv2.COLOR_BGR2GRAY), mask)
        self.keys = np.asarray([key.pt for key in keys]).reshape(-1, 2)


def static_view(before, after):
    """两帧镜头基本没动（画面中部四点位移不足 25px）；纹理不足或配准失败返回 False。"""
    try:
        matrix, _, _ = scene_registration(SceneView(before), SceneView(after))
    except (ValueError, cv2.error, np.linalg.LinAlgError):
        return False
    return same_screen(matrix)


class RejectedTargets:
    """记下的位置只用于跳过候选，不授予任何点击；清空时机由调用方按道路可能变化决定。"""

    def __init__(self):
        self.records = []

    def __len__(self):
        return len(self.records)

    def add(self, field, point, reason):
        self.records = (self.records + [dict(view=SceneView(field), point=np.asarray(point, float),
                                             reason=reason)])[-MAX_RECORDS:]

    def clear(self):
        self.records = []

    def points(self, field):
        """把能配准到当前帧的记录投影过来；返回 [(位置, 原因)]。"""
        if not self.records:
            return []
        current, found = SceneView(field), []
        for record in self.records:
            try:
                matrix, _, _ = scene_registration(record['view'], current)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                continue
            found.append((project(matrix, record['point']).reshape(2), record['reason']))
        return found

    def filter(self, field, candidates):
        """去掉离记下位置太近的候选；返回 (保留的候选, 被跳过的 [(候选, 距离, 原因)])。"""
        points = self.points(field) if candidates else []
        kept, skipped = [], []
        for item in candidates:
            near = [(float(np.linalg.norm(np.asarray(item['click'], float)-point)), reason) for point, reason in points]
            near = min(near, default=None, key=lambda value: value[0])
            if near is not None and near[0] <= REJECT_RADIUS:
                skipped.append((item, near[0], near[1]))
            else:
                kept.append(item)
        return kept, skipped
