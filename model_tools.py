# model_tools.py
import asyncio
import json
import logging
from tools.registry import discover_builtin_tools, registry

# Import and register built-in tools via the registry's own discovery flow.
discover_builtin_tools()

logger = logging.getLogger(__name__)

# 1. Core bridge functions (called by run_agent.py)
def get_tool_definitions(enabled_toolsets=None, disabled_toolsets=None, quiet_mode=False):
    """获取所有可用工具的 OpenAI Schema 定义"""
    names = set(registry.get_all_tool_names())
    return registry.get_definitions(names, quiet=quiet_mode)


def get_toolset_for_tool(tool_name: str) -> str:
    """查询工具所属的分类 (toolset)"""
    return registry.get_toolset_for_tool(tool_name)


def check_toolset_requirements() -> dict:
    """检查工具的依赖环境是否满足"""
    return registry.check_toolset_requirements()


def handle_function_call(function_name: str, function_args: dict, task_id: str = None, **kwargs) -> str:
    """将大模型的工具调用请求路由到具体的 Python 函数"""
    try:
        # 把接收到的第三个位置参数 task_id 塞进 kwargs 里传递给具体工具
        if task_id is not None:
            kwargs["task_id"] = task_id

        return registry.dispatch(function_name, function_args, **kwargs)
    except Exception as e:
        logger.error(f"Tool {function_name} failed: {e}")
        return json.dumps({"error": str(e)})


def _run_async(coro):
    """支持异步工具的执行保护"""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            return pool.submit(asyncio.run, coro).result()