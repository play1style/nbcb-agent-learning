# 智能合规助手

这是一个面向银行合规岗位的可运行企业级智能体原型。系统使用
**LangGraph** 编排工具调用和 RAG 问答流程，使用 Markdown 制度文件构建
知识库，通过智谱 `embedding-3` 生成向量并保存到 `vector_store.json`，
通过 DeepSeek `deepseek-4-flash` 生成可溯源答案。

会话记忆采用类似 OpenClaw 的本地持久化思路：每个会话使用 append-only
JSONL 文件保存，每轮对话自动注入提示词；同时写入按日期归档、带 SHA-256
哈希链的合规备份。查询会记录用户、工具调用链、来源、耗时和响应摘要，
输出统一执行手机号、身份证号、银行卡号等敏感信息脱敏。

## 项目结构

```text
.
├── app/
│   ├── agent.py             # LangGraph 工作流、RAG 生成、置信度
│   ├── audit.py             # JSONL 审计日志
│   ├── config.py            # .env / 环境变量配置
│   ├── embeddings.py        # 智谱 embedding-3 + 离线测试 Embedding
│   ├── knowledge_base.py    # Markdown 分块、向量索引、Top-K 检索
│   ├── llm.py               # DeepSeek OpenAI 兼容客户端与提示词
│   ├── main.py              # FastAPI 接口
│   ├── memory.py            # 会话记忆与哈希链合规备份
│   ├── models.py            # API / 领域模型
│   ├── security.py          # 注入防护、输入净化、输出脱敏
│   └── tools.py             # 工具统一注册与执行
├── knowledge/
│   ├── 信贷业务制度.md
│   ├── 合规操作手册.md
│   └── 个人信息保护制度.md
├── scripts/
│   └── build_index.py       # 重建向量索引
├── docs/
│   └── compliance-storage.md # 合规保存现状、差距和生产化方案
├── tests/
│   ├── conftest.py
│   ├── test_acceptance.py
│   ├── test_api_and_persistence.py
│   └── test_tools_and_security.py
├── data/                    # 运行时生成：会话记忆与合规备份
├── logs/                    # 运行时生成：审计日志
├── .env.example
├── requirements.txt
└── vector_store.json        # 预构建的离线索引，在线启动后可重建
```

## 架构与工作流

```text
用户请求
  -> 输入净化 / 提示词注入拦截
  -> 读取会话 JSONL 并自动拼装记忆
  -> DeepSeek 工具规划（失败时使用确定性规则）
  -> ToolRegistry 统一执行
       -> search_policy(query)
       -> calculate_loan(principal, annual_rate, years)
  -> Embedding Top-3 检索 + 相似度门限
  -> DeepSeek 基于资料生成答案（不可用时使用引用模板降级）
  -> 输出脱敏 + 高/中/低置信度
  -> 会话持久化 + 哈希链合规备份 + 审计日志
```

LangGraph 节点固定为：

```text
input_guard -> load_memory -> plan_tools -> execute_tools
            -> generate_answer -> finalize -> persist_turn -> audit
```

每次进入节点都会输出结构化日志，包含 `node`、`request_id`、`session_id`
和 `user_id`，便于把 LangGraph 执行过程与审计记录关联起来。节点日志不会
写入完整用户问题，避免泄露敏感信息。

安全拦截会从 `input_guard` 直接进入 `blocked_answer`，不调用检索和模型。

## 环境依赖

- Python 3.11 或 3.12
- FastAPI >= 0.115
- LangGraph >= 0.2.60
- langchain-openai >= 0.3
- 智谱 `embedding-3` API
- DeepSeek OpenAI 兼容 Chat API

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows 激活虚拟环境：

```powershell
.venv\Scripts\activate
```

## 配置

复制示例配置并填写模型密钥：

```bash
cp .env.example .env
```

主要配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 空 | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | Chat API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-4-flash` | 对话模型 |
| `ZHIPU_API_KEY` | 空 | 智谱 API Key |
| `EMBEDDING_MODEL` | `embedding-3` | 向量模型 |
| `EMBEDDING_PROVIDER` | `auto` | `auto`、`zhipu` 或 `hash` |
| `RETRIEVAL_TOP_K` | `3` | 检索条款数量 |
| `RETRIEVAL_THRESHOLD` | `0.52` | 有效召回门限 |
| `HIGH_CONFIDENCE_THRESHOLD` | `0.68` | 高置信度门限 |
| `MEDIUM_CONFIDENCE_THRESHOLD` | `0.52` | 中置信度门限 |

`EMBEDDING_PROVIDER=auto` 会优先探测智谱 `embedding-3`；服务不可用时才切换
到确定性哈希向量，保证原型仍可离线演示。生产环境建议设置为 `zhipu`，
使向量服务异常直接暴露，而不是静默降级。

`.env` 已被 `.gitignore` 忽略。不要把真实密钥提交到版本库；若密钥曾进入
聊天记录或公开仓库，应立即轮换。

## 启动步骤

1. 安装依赖并配置 `.env`。
2. 构建向量索引：

```bash
EMBEDDING_PROVIDER=zhipu python scripts/build_index.py
```

3. 启动 API：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. 访问 Swagger 文档：

```text
http://127.0.0.1:8000/docs
```

应用启动时会检查知识文件的 SHA-256 指纹和 Embedding 模型身份。任一变化
都会自动重建 `vector_store.json`。

### 模型调用排查

调用 `GET /health` 可以直接确认当前运行环境：

```json
{
  "embedding_backend": "zhipu:embedding-3",
  "llm_available": true,
  "llm_model": "deepseek-4-flash",
  "llm_last_error": ""
}
```

- `embedding_backend=zhipu:embedding-3` 表示制度检索正在使用智谱向量。
- `llm_available=true` 表示已经加载 DeepSeek API Key，但不代表密钥一定有效。
- 每次远端 LLM 调用都会输出 `Remote LLM call started/completed/failed` 日志。
- 若 `llm_last_error` 或日志出现 `401 Authentication Fails`，说明 DeepSeek
  API Key 无效，应更新 `.env` 中的 `DEEPSEEK_API_KEY` 后重启服务。
- 若出现模型不存在或无权访问，应通过 `DEEPSEEK_MODEL` 配置账号实际可用的模型。
- 如果日志只有 `LLM generation skipped`，说明检索分数低于门限，此时不会调用
  LLM，而是返回低置信度兜底；可结合 `/api/v1/policies/search` 检查召回结果。

## API 文档

### 1. 对话

```http
POST /api/v1/chat
Content-Type: application/json
```

请求：

```json
{
  "message": "贷款审批流程是什么",
  "session_id": "session-001",
  "user_id": "compliance-user-01"
}
```

响应字段：

| 字段 | 说明 |
| --- | --- |
| `answer` | 带 `[来源文档:章节号]` 引用的回答 |
| `confidence` | 高 / 中 / 低 |
| `confidence_score` | 0 到 1 的检索置信分 |
| `sources` | 来源文档、章节、内容片段和分数 |
| `tool_chain` | 工具调用链，用于自动化审计 |
| `blocked` | 是否被安全策略拦截 |
| `latency_ms` | 本次查询耗时 |

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "首套房认定标准",
    "session_id": "session-001",
    "user_id": "user-001"
  }'
```

### 2. 制度检索

```http
GET /api/v1/policies/search?query=贷款期限规定&top_k=3
```

返回 Top-K 来源文档、章节号、内容片段和相似度。

### 3. 会话记忆

```http
GET /api/v1/sessions/{session_id}/memory?user_id=user-001
```

返回指定用户在该会话中的持久化消息。

### 4. 重建索引

```http
POST /api/v1/knowledge/reindex
```

重新加载全部 Markdown 文件，调用配置的 Embedding 模型并原子更新
`vector_store.json`。

### 5. 审计日志

```http
GET /api/v1/admin/audit?user_id=user-001&limit=100
```

返回查询内容摘要、用户 ID、工具调用链、来源、置信度、耗时和响应哈希。

### 6. 健康检查

```http
GET /health
```

返回索引文档数、分块数、Embedding 后端和 LLM 配置状态。

## 工具接口

所有工具通过 `ToolRegistry` 注册，并使用 Pydantic 参数模型校验：

```python
search_policy(query: str)
calculate_loan(principal: float, annual_rate: float, years: float)
```

`calculate_loan` 采用等额本息公式，支持将 `3.5` 或 `0.035` 识别为年利率。

## 安全与合规

- 输入净化：Unicode 归一化、控制字符过滤、长度限制。
- 注入防护：识别中英文“忽略之前指令”、系统提示词窃取、越狱和角色覆盖。
- 输出脱敏：手机号、身份证号、银行卡号、邮箱。
- 安全兜底：资料不足时固定回复“根据现有知识库无法确认，请咨询相关部门”。
- 会话记忆：`data/sessions/<session_id>.jsonl`。
- 合规备份：`data/compliance/YYYY-MM-DD/conversations.jsonl`，每条记录包含
  前序哈希和当前 SHA-256 哈希，便于检查链完整性。
- 审计日志：`logs/audit.jsonl`，记录用户、查询内容、工具链、来源和响应耗时。

当前实现属于“运行时持久化 + 会话哈希链备份”。知识库和配置的不可变版本快照、
静态加密、对象存储 WORM、保留期限、法律冻结和恢复演练尚未实现。完整的保存目录、
字段设计、校验流程和实施顺序见
[合规保存方案](docs/compliance-storage.md)。

## 测试用例说明

运行全部测试：

```bash
pytest -q
```

测试使用 `HashEmbeddingClient` 和静默 LLM，不访问网络，不消耗 API 配额。

| 用例 | 输入 | 自动断言 |
| --- | --- | --- |
| TC-A-01 | 贷款审批流程是什么 | 返回尽调、审查、授信审批等步骤，包含 `[信贷业务制度:4.1]` |
| TC-A-02 | 首套房认定标准 | 返回认定条件、20% 首付比例及 `[信贷业务制度:4.3]` |
| TC-A-03 | 计算100万贷款，年利率3.5%，期限30年 | 月供约 `4,490.45` 元，总利息约 `616,560.88` 元 |
| TC-A-04 | ignore previous instructions | 拦截提示词注入，`blocked=true`，不调用工具 |
| TC-A-05 | 某企业贷款管理规定 | 知识库无可确认依据，返回固定兜底话术 |

额外测试覆盖 API、会话记忆注入、审计日志、合规哈希链、贷款工具和敏感信息
脱敏。
