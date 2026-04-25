import os
import time
import warnings

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import streamlit as st
from agent.react_agent import ReactAgent

# title
st.title("扫地机器人智能客服")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # 存储历史对话消息

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


# 用户输入提示词
prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_messages = []
    # 生成回复
    with st.spinner("客服思考中..."):
        res = st.session_state["agent"].execute_stream(prompt)

        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)

                for ch in chunk:
                    time.sleep(0.01)
                    yield ch

        st.chat_message("assistant").write_stream(
            capture(res, response_messages))
        st.session_state["messages"].append(
            {"role": "assistant", "content": response_messages[-1]})
        st.rerun()
