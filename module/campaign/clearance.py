"""按实例保存尚未清空的章节，重启后继续补清，不能跟随大厅任务跳过旧章。"""

import hashlib
import json
from pathlib import Path
import time


class ClearanceJournal:
    """持久化当前实例的清怪待办与最近验证；不同实例隔离，损坏记录不会被静默当成完成。"""

    def __init__(self, instance, root=None):
        """文件只在本地日志，实例名先散列以避免路径穿越；不保存账号或游戏凭据。"""
        root = Path(root) if root is not None else Path(__file__).resolve().parents[2] / 'log/campaign'
        identity = hashlib.sha256(str(instance).encode('utf-8')).hexdigest()[:16]
        self.path = root / f'clearance-{identity}.json'
        self.data = dict(schema=1, pending=[], completed={})
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if (self.data.get('schema') != 1 or not isinstance(self.data.get('pending'), list)
                    or not isinstance(self.data.get('completed'), dict)
                    or any(type(chapter) is not int or chapter < 1 for chapter in self.data['pending'])):
                raise ValueError('章节清怪记录无法验证，停止以避免跳过待清章节。')

    def pending(self):
        """优先补清最早未完成的章节；没有待办时返回 None，不自行向过去章节扩展范围。"""
        return min(self.data['pending'], default=None)

    def begin(self, chapter):
        """每次实际进入章节时登记待清；历史清空记录不能替代本次实时检查。"""
        if chapter not in self.data['pending']:
            self.data['pending'].append(chapter)
        self.save()

    def complete(self, chapter, proof):
        """仅接受本次完整巡检的空图证明，清除对应待办；其他未完成章节继续保留。"""
        if proof.get('state') != 'verified_empty':
            raise ValueError('缺少完整巡检证明，不能标记章节清空。')
        self.data['pending'] = [item for item in self.data['pending'] if item != chapter]
        self.data['completed'][str(chapter)] = dict(proof=proof, verified_at=time.time())
        self.save()

    def save(self):
        """原子替换待办记录，任务中断时保留上一份完整状态。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)


def pending_chapter(config):
    """持续推图入口优先读取本实例未清章节，而不是直接跟随已推进到新章的大厅任务。"""
    return ClearanceJournal(config.config_name).pending()
