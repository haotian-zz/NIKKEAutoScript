"""按实例保留设施与尝试记录；外部地图点和实测关键帧使用明确的独立坐标系。"""

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from module.campaign.vision.enemy_ring import HUD_RECTS, RingFrame, clickable_ring, scene_registration
from module.campaign.vision.surface_motion import project

MAX_VIEWS = 64
MAX_FACILITIES = 256


class FacilityUnconfirmed(RuntimeError):
    """设施身份或坐标关联不足；保留旧记录并停止，不能换一个编号再次尝试。"""
    reason = 'facility_identity_unconfirmed'


def facility_frame(field, objects):
    """只提取静态背景特征，不重复运行敌人检测，也不保留整幅截图在内存。

    HUD 和设施动画框不参与配准；关键帧最多3000个SIFT特征，后续仅在几何内点凸包内关联。
    这些坐标属于对应客户区关键帧，不能直接交给章节底图导航。
    """
    mask = np.full(field.shape[:2], 255, np.uint8)
    for left, top, right, bottom in HUD_RECTS:
        mask[top:bottom, left:right] = 0
    for item in objects:
        left, top, right, bottom = np.rint(item['box']).astype(int)
        mask[max(0, top-20):min(field.shape[0], bottom+20),
             max(0, left-20):min(field.shape[1], right+20)] = 0
    keys, descriptors = cv2.SIFT_create(nfeatures=3000, contrastThreshold=.02).detectAndCompute(
        cv2.cvtColor(field, cv2.COLOR_BGR2GRAY), mask)
    return RingFrame(np.empty((0, 2)), np.asarray([key.pt for key in keys]), descriptors,
                     np.zeros(2), [], [])


def stable_facilities(first, second):
    """登记连续两帧唯一同类同状态设施，平台跨视角的start/end变化留给身份关联处理。

    机关及屏障至少0.8，平台至少0.9；完整入屏、位移≤12px，30px内存在同类竞争时拒绝。
    已开启机关也要登记，以免将后来重新看到的设施误当作新的未使用机关。
    """
    def accepted(item):
        """只接收已知设施语义及完整客户区检测框，不将HUD或低分框纳入状态记忆。"""
        return (item['kind'] in ('ground_mechanism', 'elevator', 'barrier')
                and item['confidence'] >= (.9 if item['kind'] == 'elevator' else .8)
                and clickable_ring(item['box']))

    result = []
    for item in filter(accepted, second):
        nearby = [old for old in filter(accepted, first)
                  if old['kind'] == item['kind'] and old['state'] == item['state']
                  and np.linalg.norm(np.asarray(old['position'])-item['position']) <= 12.]
        competing = [other for other in filter(accepted, second) if other is not item
                     and other['kind'] == item['kind']
                     and np.linalg.norm(np.asarray(other['position'])-item['position']) < 30.]
        if len(nearby) == 1 and not competing:
            result.append(item)
    return result


class FacilityMemory:
    """通过有界关键帧重定位设施，跨任务保存尝试历史，绝不把历史像素当成新落点。

    身份由实例、章、难度、底图摘要共同隔离。最多64个静态关键帧和256个设施；达到容量
    停止登记而不淘汰已尝试设施。首次重新观察必须和历史关键帧配准，失败不另建重复身份。
    """

    def __init__(self, instance, identity, map_sha256=None, root=None):
        """读取本地私有记录；关键帧只含数值特征，禁止pickle，不保存账号配置或原始画面。"""
        root = Path(root) if root is not None else Path(__file__).resolve().parents[2]/'log/campaign/facilities'
        self.scope = dict(instance=hashlib.sha256(str(instance).encode()).hexdigest()[:16],
                          identity=list(identity), map_sha256=map_sha256)
        name = hashlib.sha256(json.dumps(self.scope, sort_keys=True).encode()).hexdigest()[:24]
        self.folder = root/name
        self.path = self.folder/'catalog.json'
        self.data = dict(schema=1, scope=self.scope, facilities={}, views=[], connections=[], survey={}, state_epoch=0)
        self.frames = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf8'))
            if (self.data.get('schema') != 1 or self.data.get('scope') != self.scope
                    or not isinstance(self.data.get('facilities'), dict)
                    or len(self.data['facilities']) > MAX_FACILITIES
                    or len(self.data.get('views', [])) > MAX_VIEWS):
                raise FacilityUnconfirmed('设施记录格式或范围不符，未清空历史尝试记录。')
            for view in self.data['views']:
                number = view['id']
                if type(number) is not int or not 0 <= number < MAX_VIEWS:
                    raise FacilityUnconfirmed('设施关键帧编号无效。')
                with np.load(self.folder/f'view-{number:03d}.npz', allow_pickle=False) as value:
                    keys, descriptors = value['keys'], value['descriptors']
                if keys.shape != (len(descriptors), 2) or descriptors.shape != (len(keys), 128):
                    raise FacilityUnconfirmed('设施关键帧特征不完整，未复用旧像素。')
                self.frames[number] = RingFrame(np.empty((0, 2)), keys, descriptors, np.zeros(2), [], [])
        self.current = {}
        self.unresolved = []

    def save(self):
        """原子替换小型目录；关键帧先落盘，目录只引用已完成的不可变特征文件。"""
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf8')
        temporary.replace(self.path)

    def import_annotations(self, document):
        """导入已校验底图标注及有向连接；没有实测场景绑定的地图点只供规划，不直接点击。

        只接受本记录绑定的底图摘要，不将外部静态开关标注理解为当前账号已经打开机关。
        未与场景关联的点和扫描记录分列，不能用相似位置或外观擅自配对传送入口出口。
        """
        if not self.scope['map_sha256'] or document.get('image_sha256') != self.scope['map_sha256']:
            raise FacilityUnconfirmed('外部标注与设施记录不是同一版底图。')
        for item in document.get('objects', []):
            kind = {'ground_mechanism': 'ground_mechanism', 'ground_elevator': 'elevator'}.get(item.get('category'))
            if not kind or item.get('type') != 'point':
                continue
            identifier = 'map-'+item['id']
            if identifier not in self.data['facilities']:
                self._new(identifier, kind, 'annotations')
            self.data['facilities'][identifier]['map_position'] = item['points'][0]
        for link in document.get('connections', []):
            if link.get('category') != 'elevator_connection':
                continue
            edge = dict(source='annotations', entry='map-'+link['from'], exit='map-'+link['to'],
                        directed=link['directed'], verification='provided')
            if edge['entry'] in self.data['facilities'] and edge['exit'] in self.data['facilities']:
                if edge not in self.data['connections']:
                    self.data['connections'].append(edge)
        self.save()

    def _new(self, identifier, kind, source):
        """新增有界语义记录，状态和尝试分开；不以已点击替代已激活。"""
        if len(self.data['facilities']) >= MAX_FACILITIES:
            raise FacilityUnconfirmed('设施登记达到容量，保留原有身份并停止。')
        self.data['facilities'][identifier] = dict(id=identifier, kind=kind, source=source,
                                                  observations=[], attempts=[], activated='unconfirmed')

    def observe(self, field, objects):
        """把当前两帧确认设施关联到已存关键帧，返回本帧检测与持久身份的对应。

        每次点击仍需调用方的新鲜双帧许可。身份关联允许平台外观start/end改变，只按几何和
        语义；两边凸包支持、15px距离、12px次选间隔及互为最近共同防止将不同设施合并。
        历史无法重定位时拒绝建立新身份，避免靠重新编号绕过已尝试记录。
        """
        frame = facility_frame(field, objects)
        if frame.descriptors is None or len(frame.keys) < 30:
            raise FacilityUnconfirmed('静态纹理不足，无法登记可靠设施身份。')
        matches = []
        # 顺序扫描多数只需最近一屏；重启或回访时才补查历史关键帧。
        for number in reversed(self.frames):
            try:
                matrix, source, destination = scene_registration(self.frames[number], frame)
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                continue
            matches.append((number, matrix, source, destination))
        if self.frames and not matches:
            raise FacilityUnconfirmed('当前视野未与设施关键帧可靠重定位，未创建重复设施。')
        assignments, unresolved = [], []
        for item in objects:
            options, possible = {}, set()
            for number, matrix, source, destination in matches:
                point = tuple(map(float, item['position']))
                for identifier, record in self.data['facilities'].items():
                    if record['kind'] != item['kind']:
                        continue
                    if item['kind'] == 'barrier' and record.get('last_state') != item['state']:
                        continue
                    for old in record['observations']:
                        if old['view'] != number:
                            continue
                        distance = float(np.linalg.norm(project(matrix, old['position'])-item['position']))
                        # 凸包外推仅用于否决重复建档，不授予关联或点击许可。
                        if distance <= 30.:
                            possible.add(identifier)
                        if (cv2.pointPolygonTest(source, tuple(old['position']), False) < 0
                                or cv2.pointPolygonTest(destination, point, False) < 0):
                            continue
                        options[identifier] = min(distance, options.get(identifier, float('inf')))
            ranked = sorted((distance, identifier) for identifier, distance in options.items())
            if ranked and ranked[0][0] <= 15.:
                if len(ranked) > 1 and ranked[1][0]-ranked[0][0] < 12.:
                    raise FacilityUnconfirmed('当前设施存在多个相近历史身份，未继续交互。')
                assignments.append(ranked[0][1])
            else:
                assignments.append(False if possible else None)
                if possible:
                    unresolved.append(dict(kind=item['kind'], state=item['state'], position=item['position'],
                                           possible_ids=sorted(possible), reason='identity_outside_support'))
        known = [identifier for identifier in assignments if identifier]
        if len(set(known)) != len(known):
            raise FacilityUnconfirmed('多个当前设施竞争同一历史身份，未合并或点击。')
        # 每次确认视野保留一个特征关键帧；同一屏或小幅漂移时复用旧关键帧，避免常规轮询耗尽容量。
        reuse = None
        for number, matrix, _, _ in matches:
            grid = np.float32([[400, 300], [1300, 300], [1300, 750], [400, 750]])
            if np.max(np.linalg.norm(project(matrix, grid)-grid, axis=1)) < 25.:
                reuse = number
                break
        view = reuse
        if view is None:
            if len(self.frames) >= MAX_VIEWS:
                raise FacilityUnconfirmed('设施关键帧达到容量，保留部分巡检结果，不声明全图完成。')
            view = len(self.frames)
            self.folder.mkdir(parents=True, exist_ok=True)
            temporary = self.folder/f'view-{view:03d}.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, keys=frame.keys, descriptors=frame.descriptors)
            temporary.replace(self.folder/f'view-{view:03d}.npz')
            self.frames[view] = frame
            self.data['views'].append(dict(id=view, at=time.time()))
        current = {}
        for item, identifier in zip(objects, assignments):
            if identifier is False:
                continue
            if identifier is None:
                identifier = f'scene-{len(self.data["facilities"]):04d}'
                self._new(identifier, item['kind'], 'scene_scan')
            record = self.data['facilities'][identifier]
            # 复用关键帧时将位置反投影回该帧，禁止把新帧坐标错误写成旧帧像素。
            position = np.asarray(item['position'], float)
            if reuse is not None:
                matrix = next(value[1] for value in matches if value[0] == reuse)
                position = project(np.linalg.inv(matrix), position)
            observation = dict(view=view, position=position.tolist(), state=item['state'], at=time.time())
            record['observations'] = ([old for old in record['observations'] if old['view'] != view]
                                      + [observation])
            record['last_state'] = item['state']
            if item['kind'] == 'ground_mechanism' and item['state'] == 'on':
                if record['activated'] == 'observed_off':
                    self.data['state_epoch'] = self.data.get('state_epoch', 0)+1
                record['activated'] = 'observed_on'
            elif item['kind'] == 'ground_mechanism' and item['state'] == 'off':
                record['activated'] = 'observed_off'
            current[identifier] = dict(item, facility_id=identifier)
        self.current = current
        self.unresolved = unresolved
        self.data['unresolved'] = unresolved
        self.save()
        return list(current.values())

    def eligible(self, identifier, blocked_goal):
        """只跳过已确认开启机关或针对同一受阻目标已尝试设施，不永久封禁可重复使用的传送点。"""
        record = self.data['facilities'][identifier]
        return (not (record['kind'] == 'ground_mechanism' and record['activated'] == 'observed_on')
                and not any(attempt['blocked_goal'] == blocked_goal
                            and attempt.get('state_epoch', 0) == self.data.get('state_epoch', 0)
                            for attempt in record['attempts']))

    def attempt(self, identifier, blocked_goal):
        """输入前持久登记尝试；中途取消也不会把同一个未知结果设施当成完全未试过。"""
        record = self.data['facilities'][identifier]
        record['attempts'].append(dict(blocked_goal=blocked_goal, at=time.time(), outcome='unconfirmed',
                                       gate='unconfirmed', transfer='unconfirmed',
                                       state_epoch=self.data.get('state_epoch', 0)))
        record['attempts'] = record['attempts'][-32:]
        self.save()

    def feedback(self, identifier, result):
        """只保存已测位移，不推断开门、成功传送、双向连接或胜利。"""
        attempts = self.data['facilities'][identifier]['attempts']
        if attempts:
            attempts[-1]['outcome'] = result.get('state', 'unknown')
            attempts[-1]['position_feedback'] = result
            self.save()
