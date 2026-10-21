"""章节完成后的阻塞界面流程；解锁响应与整章清理结论分别保存。"""

import time

import numpy as np

from module.campaign.vision.chapter import chapter_identity
from module.campaign.vision.chapter_completion import chapter_overlay, overlay_anchor
from module.logger import logger


class ChapterOverlayUnconfirmed(RuntimeError):
    """章节浮层或点击响应没有充分证据，停止并保留旧章待办。"""

    reason = 'chapter_overlay_unconfirmed'


class ChapterOverlayFlow:
    """依次确认MISSION CLEAR及开放探索；每条已知正文只点一次，实际换章由外层重新核对。

    所有输入经正式会话的最终帧验证与预算。仅消费已知浮层，不跳过未知提示，不更新清理簿，
    也不根据浮层消失宣称胜利、可通行或新章已经进入。
    """

    def __init__(self, session, reader):
        """绑定正式会话及OCR；每段会话独立保存已发送提示身份，避免无响应时连续重试。"""
        self.session, self.reader = session, reader
        self.sent = {}
        self.unconfirmed_since = None

    def accepted(self, evidence, field):
        """浮层证据须匹配实际普通章节；未知身份或其他章号不能沿用当前会话点击许可。"""
        if evidence is None or not evidence.get('accepted'):
            return False
        expected = self.session.expected_identity
        if expected[1] != 'normal' or chapter_identity(field, self.reader) != expected:
            return False
        return evidence['kind'] != 'chapter_completion' or evidence.get('current') == expected[0]

    @staticmethod
    def action_key(evidence):
        """以完整正文和章节关系区分连续解锁页；同为UNLOCK不能等同于同一条提示。"""
        return (evidence['kind'], evidence.get('current'), evidence.get('following'),
                evidence.get('title'), evidence.get('prompt'))

    def same_action(self, previous, current, field):
        """连续帧须为同一提示与实际控件，点位漂移最多6px；控件呼吸发光只改变边框宽度。"""
        return bool(self.accepted(current, field) and self.action_key(previous) == self.action_key(current)
                    and np.linalg.norm(np.asarray(previous['point'])-current['point']) <= 6.)

    def handle(self, field):
        """已知浮层优先于底层地图；读不清最多10秒，已点仍不消失最多8秒，不追加输入。

        返回True表示本帧仍由浮层流程消费；False才可交地图观测。点击前至少两帧确认，
        正式输入闸门还会重读第三帧。任何失败均保留证据和章节待办。
        """
        anchor = overlay_anchor(field)
        if anchor is None:
            self.unconfirmed_since = None
            return False
        now = time.monotonic()
        evidence = chapter_overlay(field, self.reader)
        key = None if evidence is None else self.action_key(evidence)
        if key in self.sent:
            if now-self.sent[key] >= 8.:
                raise ChapterOverlayUnconfirmed('章节浮层单次点击后仍未消失，停止重复输入。')
            self.session.pause(.3)
            return True
        self.session.record_evidence('chapter_overlay', field, overlay=evidence, arrays=dict(field=field))
        if not self.accepted(evidence, field):
            if self.unconfirmed_since is None:
                self.unconfirmed_since = now
            elif now-self.unconfirmed_since >= 10.:
                raise ChapterOverlayUnconfirmed('章节浮层文字、编号或地图身份未确认，未发送点击。')
            self.session.pause(.3)
            return True
        self.session.pause(.35)
        fresh = self.session.win.client()
        second = chapter_overlay(fresh, self.reader)
        if not self.same_action(evidence, second, fresh):
            if self.unconfirmed_since is None:
                self.unconfirmed_since = now
            elif now-self.unconfirmed_since >= 10.:
                raise ChapterOverlayUnconfirmed('章节浮层连续观测不一致，未发送点击。')
            return True

        def validate(latest):
            """最终输入帧重新核对同一浮层、章节和控件；既有坐标不能跨界面复用。"""
            current = chapter_overlay(latest, self.reader)
            self.session.record_evidence('chapter_overlay_guard', latest, overlay=current, arrays=dict(field=latest))
            return self.same_action(second, current, latest)

        if len(self.sent) >= 6:
            raise ChapterOverlayUnconfirmed('章节连续浮层达到六条上限，停止并保留当前画面。')
        self.session.begin_target(evidence['kind'])
        if not self.session.tap_chapter_overlay(second['point'], validate):
            raise ChapterOverlayUnconfirmed('章节浮层点击未发送，停止以保留当前界面。')
        self.sent[key] = time.monotonic()
        self.unconfirmed_since = None
        logger.info(f'章节界面已核验并单次点击：{key}；等待实际响应，旧章清理状态保持待办。')
        self.session.pause(.5)
        return True
