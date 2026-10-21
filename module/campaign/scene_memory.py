"""场景敌人观察簿：保存红圈、完整关卡身份与状态，不把旧屏幕位置当作新的点击证据。"""

import json
import time

import cv2
import numpy as np

from module.campaign.vision.detection import detect_grid
from module.campaign.vision.enemy_ring import HUD_RECTS, RING_CONFIDENCE, RingFrame, clickable_ring, scene_tracks


def ring_frame(field):
    """提取当前客户区的可点击脚圈及静态背景特征；不依赖章节底图或小地图定位。

    排除 HUD 和敌人动画区域后提取 SIFT，后续只用通过几何检验的双帧关联继承观察 ID。
    """
    rings = [item for item in detect_grid(field, ('scene_enemy_ring',))
             if item.confidence >= RING_CONFIDENCE and clickable_ring(item.box)]
    centers = np.asarray([item.center for item in rings], float).reshape(-1, 2)
    mask = np.full(field.shape[:2], 255, np.uint8)
    for left, top, right, bottom in HUD_RECTS:
        mask[top:bottom, left:right] = 0
    for x, y in np.rint(centers).astype(int):
        cv2.rectangle(mask, (x - 65, max(0, y - 140)), (x + 65, y + 35), 0, -1)
    keys, descriptors = cv2.SIFT_create(nfeatures=4000).detectAndCompute(
        cv2.cvtColor(field, cv2.COLOR_BGR2GRAY), mask)
    return RingFrame(centers, np.asarray([key.pt for key in keys]), descriptors, np.zeros(2), [], [])


class SceneEnemyMemory:
    """以经过验证的连续场景跟踪记住敌人，关卡号可将断开的观察重新归并。

    保存的是带观察帧编号的客户区坐标，不声称它们是全章世界坐标。重启后不载入旧坐标发输入；
    所有点击仍由当前两帧可见红圈提供。最多保留 512 条观察，每个敌人保留最近八次位置。
    """

    def __init__(self, path):
        """每次任务建立独立观察簿；历史文件仅供复盘，不作为新会话的定位先验。"""
        self.path = path
        self.previous, self.ids, self.records, self.sequence = None, [], {}, 0
        self.wins = []

    def observe(self, field):
        """扫描全部可见脚圈并更新轨迹；几何跟踪不可信时创建新观察，不沿用旧坐标。"""
        current = ring_frame(field)
        links = {}
        if self.previous is not None and len(self.previous.centers) and len(current.centers):
            try:
                old, tracked = scene_tracks(self.previous, current)
                for index in np.flatnonzero(tracked):
                    nearest = int(np.argmin(np.linalg.norm(self.previous.centers - old[index], axis=1)))
                    links[index] = self.ids[nearest]
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                pass
        self.sequence += 1
        ids = []
        for index, point in enumerate(current.centers):
            identifier = links.get(index)
            if identifier is None:
                identifier = f'ring-{self.sequence}-{index}'
                self.records[identifier] = dict(id=identifier, stage=None, status='pending', positions=[])
            record = self.records[identifier]
            record['positions'] = (record['positions'] + [dict(frame=self.sequence, point=point.tolist())])[-8:]
            record['seen_at'] = time.time()
            ids.append(identifier)
        self.previous, self.ids = current, ids
        while len(self.records) > 512:
            victim = next((key for key in self.records if key not in ids), None)
            if victim is None:
                break
            del self.records[victim]
        self.save()
        return [dict(self.records[key], click=point.tolist()) for key, point in zip(ids, current.centers)]

    def identify(self, identifier, stage, status='pending'):
        """用备战页核实的完整关卡号标记一个观察；不会推测点击目标与实际遇敌是否相同。"""
        if identifier is not None and identifier in self.records:
            self.records[identifier].update(stage=None if stage is None else list(stage), status=status,
                                            association='encountered_on_route')
            self.save()

    def reconsider(self, quest, victories, after_win=False):
        """任务前进后释放未来编号，胜利后复查原先受阻的道路；已胜利身份不重新排队。

        状态只影响候选排序和冷却，恢复候选仍需新帧红圈及实际备战编号验证。
        """
        for record in self.records.values():
            stage = record['stage']
            if record['status'] == 'unreachable' and after_win:
                record['status'] = 'pending'
            elif (record['status'] == 'deferred' and stage is not None and tuple(stage) not in victories
                  and quest is not None and stage[0] == quest[0] and stage[1] <= quest[1]):
                record['status'] = 'pending'
        self.save()

    def victory(self, stage):
        """记录实际关卡胜利；不据此删除被点击的目标，因为战斗可能是途中遇敌。"""
        if list(stage) not in self.wins:
            self.wins.append(list(stage))
        self.save()

    def save(self):
        """原子写入当前观察簿；文件位于本地任务日志，不包含配置或账号凭据。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        data = dict(schema=1, coordinates='client_pixels_per_frame',
                    enemies=list(self.records.values()), confirmed_wins=self.wins)
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)


def stable_candidates(first, second):
    """仅返回连续两帧唯一对应且位移不超过 12px 的新帧红圈；附近竞争目标时拒绝。

    此规则只用于下一次即时落点；跨镜头的位置记忆另由背景几何跟踪维护。
    """
    if not first or not second:
        return []
    before = np.asarray([item['click'] for item in first])
    after = np.asarray([item['click'] for item in second])
    costs = np.linalg.norm(before[:, None] - after[None, :], axis=2)
    result = []
    for j, item in enumerate(second):
        i = int(np.argmin(costs[:, j]))
        alternatives = np.r_[np.delete(costs[:, j], i), np.delete(costs[i], j)]
        if (costs[i, j] <= 12 and int(np.argmin(costs[i])) == j
                and (not len(alternatives) or alternatives.min() - costs[i, j] >= 12)):
            result.append(item)
    return result
