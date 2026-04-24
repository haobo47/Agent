## Plan: Function Calling 全量迁移 MCP

将当前 LangChain 本地 tool function-calling 方案一次性切换为 MCP（stdio 子进程）工具供给，Agent 端仅保留 MCP 工具接入，不保留本地工具兜底。迁移目标是在不改变上层交互体验（Streamlit 与 execute_stream）前提下，把工具发现、调用与协议边界统一到 MCP。

### Steps
1. 盘点并冻结现有工具契约（名称、描述、输入/输出字段、错误返回语义），作为 MCP Server 对齐基线，重点覆盖 get_weather、get_user_location、fetch_external_data。
2. 设计 MCP Server 边界与目录：新增独立 MCP server 模块（同仓或子目录），承载现有工具逻辑；明确 stdio 启动命令、健康检查与超时策略。
3. 在配置层引入 MCP 配置模型：在 agent.yaml 增加 mcp 段（transport=stdio、command、args、timeout、enabled tools 列表）；在 config_handler 增加读取与默认值处理。
4. 重构 Agent 组装入口：在 ReactAgent 中移除本地 tools 列表注入，改为 MCP client 动态拉取工具并注册到 create_agent。
5. 迁移工具实现到 MCP：把当前工具函数语义一一迁移为 MCP tools（同名优先，避免提示词与行为漂移），统一返回可序列化文本/结构；处理 fetch_external_data 目前注解为 str 但实际返回 dict 的契约不一致。
6. 处理中间件兼容：验证 monitor_tool 与 report_prompt_switch 在 MCP 工具调用路径下是否仍可拦截；若调用链变化，补充适配层保证日志与报告场景切换不丢失。
7. 一次性切换开关与清理：默认启用 MCP，删除或下线本地 FC 工具注册路径（不保留兜底）；保留最小回滚手段（Git 分支与提交点）。
8. 更新运行入口与文档：同步 app 启动依赖说明、MCP server 启动方式、排障说明（stdio 未启动、超时、协议异常）。

### Relevant Files
- d:/study/AI/Agent/agent/react_agent.py: Agent 初始化、tools 注册方式改为 MCP 来源。
- d:/study/AI/Agent/agent/tools/agent_tools.py: 现有 FC 工具契约基线，迁移后不再作为 Agent 直接 tools 列表来源。
- d:/study/AI/Agent/agent/tools/middleware.py: 工具调用链路与上下文注入兼容性验证。
- d:/study/AI/Agent/config/agent.yaml: MCP server 配置（stdio command/args/timeout/tool whitelist）。
- d:/study/AI/Agent/utils/config_handler.py: MCP 配置读取与默认值。
- d:/study/AI/Agent/app.py: 运行依赖与启动说明对齐。
- d:/study/AI/Agent/model/factory.py: 仅复核与工具协议无耦合。
- d:/study/AI/Agent/agent/tools/mcp_server.py（新增）: MCP server 工具实现与暴露入口。

### Verification
1. 工具发现验证：启动后列出 MCP tools，确认数量、名称、描述与迁移基线一致。
2. 行为一致性验证：对同一提示词回放（天气、用户位置、报告生成），比较迁移前后工具调用轨迹与输出语义。
3. 协议稳定性验证：模拟 MCP server 未启动、超时、返回非法结构，确认 Agent 侧报错可观测且不静默失败。
4. 中间件验证：确认 monitor_tool 日志仍记录到工具级事件，report_prompt_switch 场景切换生效。
5. 入口验证：app 与 react_agent 主入口都能完成一次端到端问答与工具调用。

### Decisions
- 迁移策略：一次性切换到 MCP（不走双栈并行）。
- 部署形态：本地 stdio 子进程 MCP server。
- 兜底策略：不保留本地 FC 工具兜底，严格 MCP 单路径。
- 范围包含：工具协议与注册链路改造、配置扩展、中间件兼容。
- 范围排除：本次不引入登录注册系统、不重构业务逻辑本身（仅迁移承载协议）。

### Further Considerations
1. 切换前打 pre-mcp-cutover 标签，出现阻塞可快速回滚。
2. 增补工具契约清单（name/args/return/errors），避免 MCP 与提示词漂移。
3. 后续如果要多端共享，再从 stdio 演进到 HTTP/SSE。
