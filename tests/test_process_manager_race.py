"""
ProcessManager.get_manager 并发安全回归测试。

背景：后端启动时编排器线程与 restart_processes 会并发调用 get_manager，
无锁的 check-then-act 曾导致同一实例名创建出两个 manager，后写入的空对象
覆盖了真正 fork 实例进程的对象，后端误判实例已停止、串行令牌永不授予。
"""
import queue
import logging
import threading
import time
import unittest

from module.webui import process_manager
from module.webui.process_manager import ProcessManager


class DummyManager:
    """替代 State.manager（multiprocessing.Manager），Queue 加微小延迟放大竞态窗口"""

    def Queue(self, *args, **kwargs):
        time.sleep(0.002)
        return queue.Queue(*args, **kwargs)


class GetManagerRaceTests(unittest.TestCase):
    def setUp(self):
        self._orig_manager = process_manager.State.manager
        self._orig_processes = dict(ProcessManager._processes)
        process_manager.State.manager = DummyManager()

    def tearDown(self):
        process_manager.State.manager = self._orig_manager
        ProcessManager._processes.clear()
        ProcessManager._processes.update(self._orig_processes)

    def test_concurrent_get_manager_returns_same_object(self):
        for _ in range(20):
            ProcessManager._processes.clear()
            results = []
            barrier = threading.Barrier(9)

            def worker():
                barrier.wait()
                results.append(ProcessManager.get_manager('nkas'))

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            barrier.wait()
            for t in threads:
                t.join()

            self.assertEqual(len(ProcessManager._processes), 1)
            self.assertEqual(len({id(m) for m in results}), 1)

    def test_stopped_log_consumer_has_bounded_storage_and_accepts_recent_error(self):
        """无人消费时大量调试日志不得无限占用内存，后续错误仍能替换旧条目。"""
        manager = ProcessManager('resource-test')
        for i in range(10000):
            process_manager.enqueue_renderable(manager._renderable_queue, (logging.DEBUG, str(i)))
        self.assertEqual(manager._renderable_queue.qsize(), 2048)
        process_manager.enqueue_renderable(manager._renderable_queue, (logging.ERROR, 'recent failure'))
        retained = [manager._renderable_queue.get_nowait() for _ in range(2048)]
        self.assertIn((logging.ERROR, 'recent failure'), retained)

    def test_closed_log_consumer_does_not_raise_into_game_task(self):
        """网页日志进程断开时，发送端不将管道异常传播成游戏任务失败。"""
        from unittest.mock import Mock
        target = Mock()
        target.put_nowait.side_effect = BrokenPipeError('consumer exited')
        process_manager.enqueue_renderable(target, (logging.ERROR, 'saved to file separately'))


if __name__ == '__main__':
    unittest.main()
