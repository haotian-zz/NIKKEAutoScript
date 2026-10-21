"""跨重启的清怪待办与实例隔离回归。"""

import json
from pathlib import Path
import tempfile
import unittest

from module.campaign.clearance import ClearanceJournal


class ClearanceJournalTests(unittest.TestCase):
    """主线先前进或任务中断，都不能丢失旧章的未清状态。"""

    def test_restart_keeps_old_chapter_pending(self):
        """任务已推进到下一章时仍先补旧章，新的实例记录保持隔离。"""
        with tempfile.TemporaryDirectory() as folder:
            first = ClearanceJournal('AutoPush', folder)
            first.begin(20)
            restarted = ClearanceJournal('AutoPush', folder)
            restarted.begin(21)
            self.assertEqual(restarted.pending(), 20)
            self.assertIsNone(ClearanceJournal('Other', folder).pending())

    def test_only_verified_clearance_removes_one_chapter(self):
        """不完整记录不能清除待办，完整巡检只移除已经检查过的那一章。"""
        with tempfile.TemporaryDirectory() as folder:
            journal = ClearanceJournal('AutoPush', folder)
            journal.begin(20)
            journal.begin(21)
            with self.assertRaises(ValueError):
                journal.complete(20, dict(state='incomplete'))
            self.assertEqual(journal.pending(), 20)
            journal.complete(20, dict(state='verified_empty', rows=3, pans=16))
            self.assertEqual(ClearanceJournal('AutoPush', folder).pending(), 21)
            journal.begin(20)
            self.assertEqual(journal.pending(), 20)

    def test_corrupt_journal_does_not_silently_skip_pending_work(self):
        """记录损坏或章节类型错误时停止，不回退成没有待清章节。"""
        with tempfile.TemporaryDirectory() as folder:
            journal = ClearanceJournal('AutoPush', folder)
            journal.path.parent.mkdir(parents=True, exist_ok=True)
            for content in ('{', json.dumps(dict(schema=1, pending=['20'], completed={}))):
                journal.path.write_text(content)
                with self.assertRaises(ValueError):
                    ClearanceJournal('AutoPush', folder)


if __name__ == '__main__':
    unittest.main()
