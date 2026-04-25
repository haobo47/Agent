# 智能客服AGENT

基于 LangChain ReAct Agent + RAG + Streamlit 的扫地机器人智能客服系统。

## 使用必看

本项目面向扫地机器人/扫拖一体机器人问答场景，支持知识库检索问答、在线天气与定位查询、报告生成、多轮对话与流式输出。

## 📖 项目简介

系统以 Streamlit 构建前端页面，后端基于 LangChain 搭建 ReAct Agent，并整合以下能力：

- RAG 增强检索：从本地知识库检索资料后再生成回答。
- 混合召回 + 重排序：向量召回与 BM25 融合后再重排，提高相关性。
- 高德接口能力：支持 IP 定位、城市解析、实时天气查询。
- 动态提示词切换：中间件可在普通问答与报告模式间切换。
- 临时会话记忆：同一会话内保留历史消息，实现多轮上下文对话。
- 流式输出：网页端逐字显示模型回复。

## ✨ 核心特性

| 模块 | 说明 |
| --- | --- |
| LLM | 通义千问（ChatTongyi） |
| Embedding | DashScope Embedding |
| Rerank | DashScopeRerank |
| 检索 | 向量检索 + BM25 混合召回 |
| 向量数据库 | Chroma（本地持久化） |
| Agent 框架 | LangChain ReAct Agent + LangGraph |
| 前端 | Streamlit |
| 外部服务 | 高德 REST API（天气、定位） |
| 配置管理 | YAML 配置化 |
| 文档去重 | MD5 跟踪增量入库 |

## 🏗 系统架构

```text
Streamlit(app.py)
   -> ReactAgent(agent/react_agent.py)
      -> Tools(agent/tools/agent_tools.py)
      -> Middleware(agent/tools/middleware.py)
      -> RAG(rag/rag_service.py)
         -> HybridRetriever(rag/hybrid_retriever.py)
         -> RerankService(rag/rerank_service.py)
         -> VectorStoreService(rag/vector_store.py)
            -> Chroma(chroma_db/)
```

## 📂 目录结构

```text
Agent/
├── app.py
├── agent/
│   ├── react_agent.py
│   └── tools/
│       ├── agent_tools.py
│       └── middleware.py
├── rag/
│   ├── rag_service.py
│   ├── hybrid_retriever.py
│   ├── rerank_service.py
│   └── vector_store.py
├── model/
│   └── factory.py
├── utils/
│   ├── config_handler.py
│   ├── file_handler.py
│   ├── logger_handler.py
│   ├── path_tool.py
│   └── prompt_loader.py
├── config/
│   ├── agent.yaml
│   ├── rag.yaml
│   ├── chroma.yaml
│   └── prompts.yaml
├── prompts/
│   ├── main_prompt.txt
│   ├── rag_summarize.txt
│   └── report_prompt.txt
├── data/
│   ├── *.txt
│   └── external/records.csv
├── logs/
└── .streamlit/config.toml
```

## 📦 环境依赖

- Python 3.10+
- 依赖安装：

```bash
python -m pip install -r requirements.txt
```

## ⚙️ 配置说明

### 1. 大模型与检索配置

编辑 `config/rag.yaml`：

- `chat_model_name`：聊天模型
- `embedding_model_name`：向量模型
- `retrieval`：混合召回参数
- `rerank`：重排模型参数

可按需补充：

- `dashscope_api_key`：DashScope Key（如未通过环境变量注入）

### 2. 高德接口配置

编辑 `config/agent.yaml`：

- `gaode_key`：高德 Web 服务 Key
- `gaode_base_url`：默认 `https://restapi.amap.com`
- `gaode_timeout`：接口超时时间

### 3. 向量库配置

编辑 `config/chroma.yaml`：

- `persist_directory`
- `chunk_size` / `chunk_overlap`
- `data_path`
- `allow_knowledge_file_type`

## 🚀 快速开始

1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

2. 配置 API Key

- 配置 DashScope Key（环境变量或 `config/rag.yaml`）
- 配置高德 Key（`config/agent.yaml`）

3. 启动应用

```bash
streamlit run app.py
```

默认访问：`http://localhost:8501`

## 💬 使用方式

- 产品问答：咨询故障排查、维护保养、选购建议等。
- 天气与定位：如“我现在所在城市天气怎么样”。
- 报告生成：如“帮我生成我的使用报告”。

## 🛠 工具列表

- `rag_summarize`：知识库检索总结
- `get_weather`：天气查询
- `get_user_location`：IP 定位
- `get_user_id`：用户 ID
- `get_current_month`：当前月份
- `fetch_external_data`：读取外部记录
- `fill_context_for_report`：触发报告模式

## 🔄 中间件机制

- `monitor_tool`：工具调用监控
- `log_before_model`：模型调用前日志
- `report_prompt_switch`：动态切换系统提示词

## 📋 日志说明

日志输出到 `logs/`，同时打印到控制台，便于排查工具调用和模型推理流程。

## 📄 许可证

本项目用于学习与研究。涉及第三方平台服务时，请遵守对应服务条款。
