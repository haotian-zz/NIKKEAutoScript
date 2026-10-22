"""受阻后从小队附近向外螺旋移动镜头，寻找新鲜双帧确认的关闭地面机关。

实录 2026-10-21（21-15）：光幕挡住主线敌人，门旁平台是单向到达端，点两次都不传送；
真正的开关在同一连通区南侧崖尖，镜头向南约四屏、向东两屏才进入画面，点一次后游戏
原生寻路走过去踩下，镜头随即切到原门且光幕消失。原有±360/240四次换视角看不到它。

镜头移动只用于观察；找到后由调用方点击并按实际结果复核。目录记录尽力而为，目录无法
重定位时不阻止本次新鲜检测，也不以新编号覆盖旧设施的尝试历史。
"""

import time

import numpy as np

from module.campaign.facility_memory import FacilityUnconfirmed, stable_facilities
from module.campaign.navigation import controls as goto
from module.campaign.navigation.camera_navigation import pan_scene
from module.campaign.vision.objects import detect_objects
from module.campaign.vision.perception import squad_arrow
from module.logger import logger

# 拖动手势向量（手指方向，镜头反向移动）；实测600px竖拖约半屏，边界处会被游戏夹住。
HORIZONTAL_DRAG = 700.
VERTICAL_DRAG = 520.


def spiral_drags(rings=4):
    """返回以起点为中心逐圈外扩的手势序列：先南、东，再北、西，每两段边长加一。"""
    directions = ((0., -VERTICAL_DRAG), (-HORIZONTAL_DRAG, 0.), (0., VERTICAL_DRAG), (HORIZONTAL_DRAG, 0.))
    drags, length = [], 1
    for leg in range(rings*2):
        drags += [directions[leg % 4]]*length
        if leg % 2:
            length += 1
    return [np.asarray(drag) for drag in drags]


def off_switches(first, second):
    """两帧同位置、完整入屏、无同类竞争的关闭地面机关；开启机关和平台不在此列。"""
    return [item for item in stable_facilities(first, second)
            if item['kind'] == 'ground_mechanism' and item['state'] == 'off']


class SwitchSearch:
    """有界螺旋镜头搜索；任何真实备战弹窗立即交回战斗流程，不把找到当作开门。"""

    def __init__(self, runner, catalog=None, pans=20, seconds=100., skip=()):
        """skip是本受阻点已尝试过且未解决的搜索键，避免同状态下重复点同一个机关。"""
        self.runner, self.session, self.catalog = runner, runner.session, catalog
        self.drags = spiral_drags()[:pans]
        self.seconds, self.skip = seconds, set(skip)
        self.views, self.pans, self.catalog_errors = 0, 0, 0

    def look(self, step):
        """取两帧检测关闭机关并登记目录；返回(状态, 候选, 新鲜帧)。"""
        field = self.runner.map_frame()
        if goto.battle_popup_score(field) > .8:
            return 'entry', None, field
        first = detect_objects(field, ('ground_mechanism', 'elevator', 'barrier'))
        self.session.pause(.25)
        fresh = self.runner.map_frame()
        if goto.battle_popup_score(fresh) > .8:
            return 'entry', None, fresh
        second = detect_objects(fresh, ('ground_mechanism', 'elevator', 'barrier'))
        self.views += 1
        registered = []
        if self.catalog is not None:
            try:
                registered = self.catalog.observe(fresh, stable_facilities(first, second))
            except FacilityUnconfirmed:
                self.catalog_errors += 1
        candidates = []
        for item in off_switches(first, second):
            key = (step, *np.round(np.asarray(item['position'])/80.).astype(int).tolist())
            known = [entry['facility_id'] for entry in registered if entry['kind'] == item['kind']
                     and np.linalg.norm(np.asarray(entry['position'])-item['position']) < 1.]
            if key in self.skip or (known and not self.catalog.eligible(
                    known[0], 'quest:'+str(self.runner.quest))):
                continue
            candidates.append(dict(item, search_key=key, **({'facility_id': known[0]} if known else {})))
        self.session.record_evidence('switch_search_view', fresh, step=step, detections=second,
                                     candidates=candidates, arrays=dict(field=fresh))
        self.session.preview(fresh)
        if not candidates:
            return 'none', None, fresh
        arrow = squad_arrow(fresh)
        anchor = np.asarray(goto.VIEWPORT_PIXEL if arrow is None else arrow, float)
        return 'found', min(candidates, key=lambda item: np.linalg.norm(
            np.asarray(item['position'])-anchor)), fresh

    def run(self):
        """原地先看一屏，再按螺旋逐步拖镜头；返回dict(state=found/entry/exhausted, ...)。"""
        started = time.monotonic()
        logger.info(f'目标受阻：开始螺旋寻找关闭机关，最多 {len(self.drags)} 次镜头移动。')
        for step in range(len(self.drags)+1):
            self.session.check()
            if time.monotonic()-started > self.seconds:
                break
            if step:
                field = self.runner.map_frame()
                arrow = squad_arrow(field)
                ground = None if arrow is None else np.asarray(arrow)+[0., goto.ARROW_GROUND_OFFSET]
                self.session.begin_target('switch_search_pan')
                pan_scene(self.session, self.drags[step-1], squad_ground=ground)
                self.pans += 1
            state, target, field = self.look(step)
            if state == 'entry':
                return dict(state='entry', field=field, views=self.views, pans=self.pans)
            if state == 'found':
                logger.info(f"螺旋搜索第 {step} 屏发现关闭机关 {target['position']}"
                            f"（{target['confidence']:.2f}），交由原生寻路前往；开门须另行复核。")
                return dict(state='found', target=target, field=field, views=self.views, pans=self.pans)
        logger.info(f'螺旋搜索未发现可用关闭机关：{self.views} 屏，{self.pans} 次镜头移动，'
                    f'目录无法重定位 {self.catalog_errors} 次。')
        return dict(state='exhausted', views=self.views, pans=self.pans, catalog_errors=self.catalog_errors)
