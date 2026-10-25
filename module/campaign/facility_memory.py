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
# 每个设施保留最近几次视野观测；更早的视野不再锚定它，可作为桥接帧淘汰。
MAX_OBSERVATIONS = 4
# 先按特征投票挑出最可能重叠的几个关键帧再做严格配准；都配不上才补查其余关键帧。
# 实测 2026-10-22（22章64个关键帧）：逐个严格配准约3秒，投票约0.06秒，票数前几名基本都是真重叠。
CANDIDATE_VIEWS = 8
# 另外总带上票数最高的几个锚点帧：设施身份和需手动关卡只能经锚点帧投影，候选全是桥接帧时
# 会认不出已登记的位置（实录 2026-10-22，22-36 同一个 EX 记了两次）。
ANCHOR_CANDIDATES = 4
VOTE_SAMPLE = 600
VOTE_DISTANCE = 200.


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


def accepted_facility(item):
    """只接收已知设施语义及完整客户区检测框，不将HUD或低分框纳入状态记忆。"""
    return (item['kind'] in ('ground_mechanism', 'elevator', 'barrier')
            and item['confidence'] >= (.9 if item['kind'] == 'elevator' else .8)
            and clickable_ring(item['box']))


def same_screen(matrix):
    """配准矩阵把画面中部四点移动不足25px：视为同一屏或小幅漂移，可复用该关键帧。"""
    grid = np.float32([[400, 300], [1300, 300], [1300, 750], [400, 750]])
    return np.max(np.linalg.norm(project(matrix, grid)-grid, axis=1)) < 25.


def voted_facilities(frames, minimum=2):
    """同一镜头下多帧投票：同类同状态、≤12px 的检测出现在至少 minimum 帧即接受。

    实录 2026-10-21（21-32）：烟雾里的关闭机关三帧漏检一帧，严格两帧连续命中只有约一半
    概率保留，螺旋或巡检扫过也会丢失。门槛和完整入屏要求不变；返回该设施最后一次出现的
    检测，任一帧内 30px 内存在同类竞争时拒绝。
    """
    frames = [[item for item in frame if accepted_facility(item)] for frame in frames]
    result = []
    for index in range(len(frames)-1, -1, -1):
        for item in frames[index]:
            def same(other):
                return (other['kind'] == item['kind'] and other['state'] == item['state']
                        and np.linalg.norm(np.asarray(other['position'])-item['position']) <= 12.)
            if any(same(old) for old in result):
                continue
            hits = sum(any(same(other) for other in frame) for frame in frames)
            competing = False
            for frame in frames:
                near = [other for other in frame if other['kind'] == item['kind']
                        and np.linalg.norm(np.asarray(other['position'])-item['position']) < 30.]
                if sum(map(same, near)) > 1 or not all(map(same, near)):
                    competing = True
            if hits >= minimum and not competing:
                result.append(item)
    return result


def stable_facilities(first, second):
    """登记连续两帧唯一同类同状态设施，平台跨视角的start/end变化留给身份关联处理。

    机关及屏障至少0.8，平台至少0.9；完整入屏、位移≤12px，30px内存在同类竞争时拒绝。
    已开启机关也要登记，以免将后来重新看到的设施误当作新的未使用机关。
    """
    accepted = accepted_facility
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

    身份由实例、章、难度、底图摘要共同隔离。最多64个静态关键帧和256个设施；首次重新观察
    必须和历史关键帧配准，失败不另建重复身份。

    关键帧分两类（用户 2026-10-22 指出“顺手记下的背景帧”不该占满容量）：锚点帧被某个设施的
    观测引用，设施身份靠它重定位，永不淘汰；桥接帧只是移动途中保持配准连续的背景，满了按最久
    未命中先淘汰。实录 2026-10-22（22-36）：64个关键帧里40个是桥接帧，容量耗尽后全章巡检一屏
    都没登记，用户自己找到的开关目录里没有。
    """

    def __init__(self, instance, identity, map_sha256=None, root=None):
        """读取本地私有记录；关键帧只含数值特征，禁止pickle，不保存账号配置或原始画面。"""
        root = Path(root) if root is not None else Path(__file__).resolve().parents[2]/'log/campaign/facilities'
        self.scope = dict(instance=hashlib.sha256(str(instance).encode()).hexdigest()[:16],
                          identity=list(identity), map_sha256=map_sha256)
        name = hashlib.sha256(json.dumps(self.scope, sort_keys=True).encode()).hexdigest()[:24]
        self.folder = root/name
        self.path = self.folder/'catalog.json'
        # 设施位置与关键帧和账号无关，按章节、难度、底图共享；尝试记录与开关状态属于账号，不共享。
        shared_scope = dict(identity=list(identity), map_sha256=map_sha256)
        self.shared = root/'shared'/hashlib.sha256(json.dumps(shared_scope, sort_keys=True).encode()).hexdigest()[:24]
        self.data = dict(schema=1, scope=self.scope, facilities={}, views=[], connections=[], survey={}, state_epoch=0)
        self.frames = {}
        if not self.path.exists():
            self._seed_shared()
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
        self.expected = []
        self.manual_projected = []
        self.unresolved = []
        self.last_bridge_at = 0.
        self.vote_index = None

    def save(self):
        """原子替换小型目录；关键帧先落盘，目录只引用已完成的不可变特征文件。"""
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf8')
        temporary.replace(self.path)
        self._publish_shared()

    def _shared_document(self):
        """共享副本只含设施类别、来源和逐视野观测位置，去掉尝试、激活状态与未决记录。"""
        facilities = {identifier: dict(id=identifier, kind=record['kind'], source=record['source'],
                                       observations=[dict(view=old['view'], position=old['position'])
                                                     for old in record['observations']],
                                       **({key: record[key] for key in ('map_position', 'map_source', 'map_samples')
                                           if key in record}),
                                       # 屏障的状态是部件外观（光幕/立柱），身份关联要用。
                                       **({'last_state': record['last_state']}
                                          if record['kind'] == 'barrier' and 'last_state' in record else {}))
                      for identifier, record in self.data['facilities'].items()}
        return dict(schema=1, scope=dict(identity=self.scope['identity'], map_sha256=self.scope['map_sha256']),
                    facilities=facilities, views=self.data['views'],
                    connections=self.data['connections'], survey=self.data.get('survey', {}))

    def _publish_shared(self):
        """本实例视野不少于共享副本时整体替换；不同实例编号互不兼容，不做逐条合并。"""
        path = self.shared/'catalog.json'
        try:
            if path.exists() and len(json.loads(path.read_text(encoding='utf8')).get('views', [])) > len(self.data['views']):
                return
            self.shared.mkdir(parents=True, exist_ok=True)
            for view in self.data['views']:
                source = self.folder/f"view-{view['id']:03d}.npz"
                target = self.shared/source.name
                if source.exists() and (not target.exists() or target.stat().st_size != source.stat().st_size):
                    target.write_bytes(source.read_bytes())
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(self._shared_document(), ensure_ascii=False, indent=2), encoding='utf8')
            temporary.replace(path)
        except (OSError, ValueError):
            # 共享副本只是加速，失败不影响本实例记录。
            pass

    def _seed_shared(self):
        """本实例首次进入该章时以共享副本起步：位置与关键帧复用，尝试和激活状态从未确认开始。"""
        path = self.shared/'catalog.json'
        try:
            document = json.loads(path.read_text(encoding='utf8'))
        except (OSError, ValueError):
            return
        if (document.get('schema') != 1 or document.get('scope') != dict(
                identity=self.scope['identity'], map_sha256=self.scope['map_sha256'])):
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        for view in document.get('views', []):
            name = f"view-{view['id']:03d}.npz"
            if not (self.shared/name).exists():
                return
            (self.folder/name).write_bytes((self.shared/name).read_bytes())
        self.data.update(views=document['views'], connections=document.get('connections', []),
                         survey=document.get('survey', {}),
                         facilities={identifier: dict(record, observations=[dict(old, state='unknown', at=0.)
                                                                           for old in record['observations']],
                                                      attempts=[], activated='unconfirmed')
                                     for identifier, record in document['facilities'].items()})
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

    def observe(self, field, objects, background=False):
        """把当前两帧确认设施关联到已存关键帧，返回本帧检测与持久身份的对应。

        每次点击仍需调用方的新鲜双帧许可。身份关联允许平台外观start/end改变，只按几何和
        语义；两边凸包支持、15px距离、12px次选间隔及互为最近共同防止将不同设施合并。
        历史无法重定位时拒绝建立新身份，避免靠重新编号绕过已尝试记录。
        background=True只追加背景关键帧，不改变设施状态、当前检测及未决身份；objects须为空。
        """
        if background and objects:
            raise ValueError('背景桥接不能混入设施观测。')
        frame = facility_frame(field, objects)
        if frame.descriptors is None or len(frame.keys) < 30:
            raise FacilityUnconfirmed('静态纹理不足，无法登记可靠设施身份。')
        matches = []
        for group in self._vote_order(frame):
            for number in group:
                try:
                    matrix, source, destination = scene_registration(self.frames[number], frame)
                except (ValueError, cv2.error, np.linalg.LinAlgError):
                    continue
                matches.append((number, matrix, source, destination))
                # 背景桥接只需确认当前视野已有关键帧：命中同屏关键帧即停，不再逐个比对全部历史
                # （实测 2026-10-22：22章46个关键帧逐个配准约1.5秒，每帧地图轮询都付一次）。
                if background and same_screen(matrix):
                    break
            if matches:
                break
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
                    # 实录 2026-10-25（HARD 9）：目录里同一机关登记过两次，设施巡检遇到它就整轮停止。
                    # 身份不明只让这一个设施本帧不可交互（不进 current），其他设施和巡检照常。
                    assignments.append(False)
                    unresolved.append(dict(kind=item['kind'], state=item['state'], position=item['position'],
                                           possible_ids=[identifier for _, identifier in ranked[:2]],
                                           reason='ambiguous_identity'))
                    continue
                assignments.append(ranked[0][1])
            else:
                assignments.append(False if possible else None)
                if possible:
                    unresolved.append(dict(kind=item['kind'], state=item['state'], position=item['position'],
                                           possible_ids=sorted(possible), reason='identity_outside_support'))
        known = [identifier for identifier in assignments if identifier]
        contested = {identifier for identifier in known if known.count(identifier) > 1}
        for index, identifier in enumerate(assignments):
            # 多个当前设施竞争同一历史身份：都不合并、本帧都不可交互，其余设施照常登记。
            if identifier in contested:
                assignments[index] = False
                item = objects[index]
                unresolved.append(dict(kind=item['kind'], state=item['state'], position=item['position'],
                                       possible_ids=[identifier], reason='contested_identity'))
        # 每次确认视野保留一个特征关键帧；同一屏或小幅漂移时复用旧关键帧，避免常规轮询耗尽容量。
        reuse = next((number for number, matrix, _, _ in matches if same_screen(matrix)), None)
        if reuse is not None:
            # 命中即刷新，淘汰桥接帧时按最久未命中先淘汰；只改内存，下次保存目录时一并写入。
            next(item for item in self.data['views'] if item['id'] == reuse)['at'] = time.time()
        if background and reuse is not None:
            # 同屏背景不新增关键帧，记录没有变化，不必重写目录。
            return []
        view = reuse
        if view is None:
            view = self._free_view({number for number, *_ in matches})
            self.folder.mkdir(parents=True, exist_ok=True)
            temporary = self.folder/f'view-{view:03d}.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, keys=frame.keys, descriptors=frame.descriptors)
            temporary.replace(self.folder/f'view-{view:03d}.npz')
            self.frames[view] = frame
            self.data['views'].append(dict(id=view, at=time.time()))
        if background:
            self.save()
            return []
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
                                      + [observation])[-MAX_OBSERVATIONS:]
            record['last_state'] = item['state']
            if item['kind'] == 'ground_mechanism' and item['state'] == 'on':
                if record['activated'] == 'observed_off':
                    self.data['state_epoch'] = self.data.get('state_epoch', 0)+1
                record['activated'] = 'observed_on'
            elif item['kind'] == 'ground_mechanism' and item['state'] == 'off':
                record['activated'] = 'observed_off'
            current[identifier] = dict(item, facility_id=identifier)
        self.current = current
        self.expected = self._expected(matches, current)
        self.manual_projected = self._manual_projected(matches)
        self.unresolved = unresolved
        self.data['unresolved'] = unresolved
        self.save()
        return list(current.values())

    def _vote_order(self, frame):
        """返回(候选关键帧, 其余关键帧)：每个关键帧抽样特征建一个近邻索引，当前画面特征投票排序。

        投票只决定严格配准的先后，不授予关联；关键帧不多时直接按新到旧全部配准。
        """
        numbers = list(reversed(self.frames))
        if len(numbers) <= CANDIDATE_VIEWS:
            return numbers, []
        if self.vote_index is None or self.vote_index[0] != tuple(numbers):
            owners, samples = [], []
            for number in numbers:
                descriptors = self.frames[number].descriptors
                sample = descriptors[::max(1, len(descriptors)//VOTE_SAMPLE)][:VOTE_SAMPLE]
                samples.append(sample)
                owners.append(np.full(len(sample), number))
            index = cv2.flann_Index(np.vstack(samples).astype(np.float32), dict(algorithm=1, trees=4))
            self.vote_index = (tuple(numbers), np.concatenate(owners), index)
        _, owners, index = self.vote_index
        nearest, distance = index.knnSearch(frame.descriptors.astype(np.float32), 1, params=dict(checks=32))
        votes = np.bincount(owners[nearest[distance[:, 0] < VOTE_DISTANCE**2, 0]], minlength=max(numbers)+1)
        ranked = sorted(numbers, key=lambda number: -votes[number])
        anchors = self.anchors()
        first = ranked[:CANDIDATE_VIEWS] + [number for number in ranked[CANDIDATE_VIEWS:]
                                            if number in anchors and votes[number]][:ANCHOR_CANDIDATES]
        return first, [number for number in ranked if number not in first]

    def anchors(self):
        """被设施观测引用的锚点关键帧编号；其余都是可淘汰的桥接帧。"""
        return {old['view'] for record in self.data['facilities'].values() for old in record['observations']}

    def _free_view(self, matched):
        """返回新关键帧可用的编号；满了淘汰一个最久未命中的桥接帧，只有锚点帧时拒绝登记。

        优先淘汰这次没配准上的桥接帧：刚配准上的帧连着当前视野，淘汰它会让下一屏更难重定位。
        """
        used = {item['id'] for item in self.data['views']}
        if len(used) < MAX_VIEWS:
            return min(set(range(MAX_VIEWS))-used)
        anchors = self.anchors()
        bridges = [item for item in self.data['views'] if item['id'] not in anchors]
        if not bridges:
            raise FacilityUnconfirmed('设施关键帧全部是锚点帧，保留部分巡检结果，不声明全图完成。')
        number = min(bridges, key=lambda item: (item['id'] in matched, item.get('at', 0.)))['id']
        (self.folder/f'view-{number:03d}.npz').unlink(missing_ok=True)
        self.frames.pop(number, None)
        self.data['views'] = [item for item in self.data['views'] if item['id'] != number]
        return number

    def needs_map(self, items):
        """本帧已登记设施里还没有底图坐标的；外部标注自带坐标，不需要补。"""
        return [item for item in items if 'map_position' not in self.data['facilities'][item['facility_id']]]

    def place(self, items, to_map):
        """把本帧设施的客户区位置换成底图坐标，只作“去哪里看”的估计，不授予点击许可。

        to_map 把客户区像素换到底图像素。已有坐标的设施只返回本次估计与记录的偏差（标注坐标
        不被改写，偏差用来核对换算精度）；实测估计按最多8次取平均，减少单次配准误差。
        """
        errors = []
        for item in items:
            record = self.data['facilities'][item['facility_id']]
            point = np.asarray(to_map(item['position']), float)
            source = record.get('map_source', 'annotations' if 'map_position' in record else 'viewport_estimate')
            if 'map_position' in record:
                errors.append(dict(facility_id=item['facility_id'], source=source,
                                   error=round(float(np.linalg.norm(point-record['map_position'])), 1)))
            if source != 'viewport_estimate':
                continue
            count = min(record.get('map_samples', 0), 7)
            old = np.asarray(record.get('map_position', point), float)
            record['map_position'] = ((old*count+point)/(count+1)).tolist()
            record['map_source'], record['map_samples'] = 'viewport_estimate', count+1
        self.save()
        return errors

    def _expected(self, matches, current):
        """已知仍关闭的地面机关投影到本帧的位置，作为“去哪里看”的线索；本帧已检测到的不列入。

        线索只来自可靠配准的历史关键帧，不授予点击许可：调用方须把镜头移过去后用新鲜检测确认。
        """
        hints = []
        for identifier, record in self.data['facilities'].items():
            if (record['kind'] != 'ground_mechanism' or record.get('activated') != 'observed_off'
                    or identifier in current):
                continue
            for number, matrix, _, _ in matches:
                old = next((item for item in record['observations'] if item['view'] == number), None)
                if old is not None:
                    hints.append(dict(facility_id=identifier, kind='ground_mechanism', state='off',
                                      position=project(matrix, old['position']).tolist()))
                    break
        return hints

    def _manual_projected(self, matches):
        """已记下的需手动关卡（EX）位置投影到本帧；只用于跳过目标，不授予任何点击。"""
        result = []
        for identifier, record in self.data['facilities'].items():
            if record['kind'] != 'manual_entry':
                continue
            for number, matrix, _, _ in matches:
                old = next((item for item in record['observations'] if item['view'] == number), None)
                if old is not None:
                    result.append(dict(facility_id=identifier, stage=record.get('stage'),
                                       position=project(matrix, old['position']).tolist()))
                    break
        return result

    def has_manual_entries(self):
        """本章是否记过需手动关卡位置。"""
        return any(record['kind'] == 'manual_entry' for record in self.data['facilities'].values())

    def remember_manual(self, field, position, stage):
        """把需手动关卡（EX）的位置作为设施登记到关键帧坐标系，换镜头后仍能认出。

        实录 2026-10-22（22-36）：三个 EX 敌人每次关闭备战后换了镜头就是新编号，下一轮又被选中；
        展开地图定位在 EX 旁常常不通过，因此用设施目录的关键帧配准记位置。position 是关闭备战
        时小队脚下的客户区位置（小队触发备战时站在该敌人旁）。
        """
        x, y = map(float, position)
        item = dict(kind='manual_entry', state='manual', confidence=1., position=[x, y],
                    box=[x-30., y-30., x+30., y+30.])
        registered = self.observe(field, [item])
        for entry in registered:
            self.data['facilities'][entry['facility_id']]['stage'] = list(stage)
        self.save()
        return registered

    def track_background(self, field, force=False):
        """保留移动途中的静态桥接关键帧，不修改设施身份、状态或尝试记录。

        普通地图帧最多每1.5秒处理一次；显式镜头寻位使用force确保每步被跟踪。只接受
        与旧图有可靠重叠的背景，失败返回False，不建立新身份、不删除历史。True在非force
        模式也可能代表本次被限频，不能作为通行或设施重新定位许可。
        """
        now = time.monotonic()
        if not force and now-self.last_bridge_at < 1.5:
            return True
        self.last_bridge_at = now
        try:
            self.observe(field, [], background=True)
        except (FacilityUnconfirmed, cv2.error, np.linalg.LinAlgError):
            return False
        return True

    def eligible(self, identifier, blocked_goal):
        """只跳过已确认开启机关或针对同一受阻目标已尝试设施，不永久封禁可重复使用的传送点。"""
        record = self.data['facilities'][identifier]
        return (not (record['kind'] == 'ground_mechanism' and record['activated'] == 'observed_on')
                and not any(attempt['blocked_goal'] == blocked_goal
                            and attempt.get('state_epoch', 0) == self.data.get('state_epoch', 0)
                            for attempt in record['attempts']))

    def switch_activated(self):
        """点下的关闭机关已在原位确认开启：道路可能整体改变，此前对各受阻目标的尝试都重新可用。"""
        self.data['state_epoch'] = self.data.get('state_epoch', 0)+1
        self.save()

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
