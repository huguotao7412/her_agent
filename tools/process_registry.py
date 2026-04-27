# tools/process_registry.py
import logging

logger = logging.getLogger(__name__)


class DummyProcessRegistry:
    """一个纯净的假体，用来骗过网关的初始化，剥离工作属性"""

    def __init__(self):
        # 假装没有正在排队监控的后台进程
        self.pending_watchers = []

    def has_active_for_session(self, session_key: str) -> bool:
        # 告诉网关：这个用户没有在跑什么后台任务，可以随时聊天/重置
        return False

    def recover_from_checkpoint(self) -> int:
        # 假装从崩溃中恢复了 0 个进程
        return 0

    def kill_all(self) -> None:
        # 网关关闭时调用的清理函数，什么都不用做
        pass


# 导出一个名为 process_registry 的假实例供网关导入
process_registry = DummyProcessRegistry()