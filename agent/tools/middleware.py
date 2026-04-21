from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import ModelRequest, before_model, dynamic_prompt, wrap_tool_call
from typing import Callable
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.logger_handler import logger
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from utils.prompt_loader import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(
    # 请求的数据
    request: ToolCallRequest,
    # 工具函数
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:  # 监控工具执行
    logger.info(
        f"[monitor_tool] 工具{request.tool_call['name']}被调用，输入参数为：{request.tool_call['args']}")

    try:
        res = handler(request)
        logger.info(
            f"[monitor_tool] 工具{request.tool_call['name']}执行成功")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return res
    except Exception as e:
        logger.error(
            f"[monitor_tool] 工具{request.tool_call['name']}执行失败，错误信息: {str(e)}", exc_info=True)
        raise e


@before_model
def log_before_model(
    state: AgentState,  # agent智能体状态记录
    runtime: Runtime,  # 记录执行过程中的上下文信息
):  # 模型执行前输出日志
    logger.info(
        f"[log_before_model] 模型即将执行，带有{len(state['messages'])}条消息")
    logger.debug(
        f"[log_before_model] {type(state['messages'][-1]).__name__} | 消息内容为：{state['messages'][-1].content.strip()}")
    return None


@dynamic_prompt              # 每次在提示词生成前调用
def report_prompt_switch(request: ModelRequest):  # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:  # 报告生成场景
        print("***********************\n")
        return load_report_prompts()
    else:  # 默认场景
        return load_system_prompts()
