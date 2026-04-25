from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location,
                                     get_user_id, get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=None,
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str, history_messages: list[dict[str, str]] | None = None):
        # 默认兼容旧调用；传入历史消息时，使用完整会话作为模型输入。
        if history_messages:
            input_messages = [
                {
                    "role": str(msg.get("role", "")).strip(),
                    "content": str(msg.get("content", "")),
                }
                for msg in history_messages
                if msg.get("role") and msg.get("content")
            ]
        else:
            input_messages = [
                {"role": "user", "content": query},
            ]

        input_dict = {
            "messages": input_messages
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values",
                                       context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip()+"\n"


if __name__ == "__main__":
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
