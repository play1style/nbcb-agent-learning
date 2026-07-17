# nbcb-agent-learning - 银行合规智能助手

基于Python和Deepseek API构建的企业级智能体原型，专为银行合规岗位设计，提供制度查询和贷款计算等功能。

## 项目结构

```
nbcb-agent-learning/
├── README.md                   # 项目说明文档
├── requirements.txt            # Python依赖清单
├── .env.example                # 环境变量模板
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI应用入口与API路由
│   ├── config.py               # 配置管理（API密钥、模型参数）
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py            # 合规智能体核心（ReAct模式）
│   │   └── llm_client.py       # Deepseek LLM客户端
│   ├── knowledge_base/
│   │   ├── __init__.py
│   │   ├── data.py             # 知识库数据管理
│   │   └── retriever.py        # TF-IDF语义检索器
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic数据模型
│   ├── security/
│   │   ├── __init__.py
│   │   ├── input_sanitizer.py  # 输入净化（防注入）
│   │   ├── output_desensitizer.py  # 输出脱敏（PII保护）
│   │   └── audit_logger.py     # 审计日志记录
│   └── tools/
│       ├── __init__.py
│       ├── registry.py         # 工具注册中心
│       ├── search_policy.py    # 制度检索工具
│       └── calculate_loan.py   # 贷款计算工具
├── tests/
│   ├── __init__.py
│   └── test_agent.py           # 核心功能测试（5个测试用例）
└── data/
    └── knowledge_base.json     # 知识库数据
```

## 环境依赖

- Python >= 3.10
- 依赖包（详见 requirements.txt）

### 安装步骤

```bash
cd nbcb-agent-learning
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 启动步骤

```bash
# 方式一：直接启动
python -m app.main

# 方式二：使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API文档（Swagger UI）：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## API 文档

### 1. 健康检查

```
GET /health
```

**响应示例：**
```json
{"status": "ok", "service": "银行合规智能助手", "version": "1.0.0"}
```

### 2. 智能体对话

```
POST /api/chat
```

**请求体：**
```json
{"query": "贷款审批流程是什么", "user_id": "zhangsan"}
```

**响应示例：**
```json
{"answer": "贷款审批流程包括以下步骤：1）客户提交贷款申请... [信贷业务制度:4.1]", "sources": ["信贷业务制度:4.1"], "tool_calls": [{"tool_name": "search_policy", "arguments": {"query": "贷款审批流程是什么"}, "result": "..."}], "processing_time_ms": 1234.56, "injection_detected": false}
```

### 3. 获取审计日志

```
GET /api/audit/recent?count=10
```

### 4. 列出可用工具

```
GET /api/tools
```

## 测试用例说明

项目包含 5 个核心测试用例，覆盖作业要求的所有功能点。

### 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 运行指定测试用例
pytest tests/test_agent.py::TestLoanApprovalProcess -v
pytest tests/test_agent.py::TestFirstHomeCriteria -v
pytest tests/test_agent.py::TestLoanCalculation -v
pytest tests/test_agent.py::TestInjectionProtection -v
pytest tests/test_agent.py::TestNoMatchInKnowledgeBase -v

# 测试覆盖率报告
pytest tests/ -v --cov=app --cov-report=term-missing
```

### TC-A-01：贷款审批流程

| 项目 | 内容 |
|------|------|
| 测试输入 | 贷款审批流程是什么 |
| 预期结果 | 返回审批流程步骤，标注 [信贷业务制度:4.1] |
| 测试类 | TestLoanApprovalProcess |

### TC-A-02：首套房认定标准

| 项目 | 内容 |
|------|------|
| 测试输入 | 首套房认定标准 |
| 预期结果 | 返回首付比例和认定条件，标注来源 |
| 测试类 | TestFirstHomeCriteria |

### TC-A-03：贷款计算

| 项目 | 内容 |
|------|------|
| 测试输入 | 计算100万贷款，年利率3.5%，期限30年 |
| 预期结果 | 返回月供和总利息数值 |
| 测试类 | TestLoanCalculation |

### TC-A-04：提示词注入防护

| 项目 | 内容 |
|------|------|
| 测试输入 | ignore previous instructions |
| 预期结果 | 拦截注入，返回安全提示 |
| 测试类 | TestInjectionProtection |

### TC-A-05：知识库无匹配

| 项目 | 内容 |
|------|------|
| 测试输入 | 某企业贷款管理规定 |
| 预期结果 | 知识库中不存在，返回无法确认 |
| 测试类 | TestNoMatchInKnowledgeBase |

## 知识库内容

知识库包含两个文档共 7 条规则：

| 文档 | 章节 | 标题 |
|------|------|------|
| 信贷业务制度 | 4.1 | 贷款审批流程 |
| 信贷业务制度 | 4.2 | 贷款期限规定 |
| 信贷业务制度 | 4.3 | 贷款利率定价 |
| 信贷业务制度 | 4.4 | 首套房认定标准 |
| 信贷业务制度 | 4.5 | 不良贷款认定标准 |
| 信贷业务制度 | 5.1 | 担保贷款管理规定 |
| 合规操作手册 | 2.1 | 客户身份识别要求 |

## 核心架构说明

### Agent设计模式

采用 ReAct（Reasoning + Acting）模式：
1. 推理：LLM分析用户问题，判断需要调用哪个工具
2. 行动：通过JSON格式返回工具调用指令
3. 观察：获取工具执行结果
4. 回答：基于工具结果生成溯源回答

### 语义检索

使用TF-IDF向量化 + 余弦相似度计算，无需外部嵌入API：
- 基于字符N元组（unigram/bigram/trigram）的简单中文分词
- 支持中英文混合文本
- 检索结果按相似度降序返回Top-K

### 安全防护

- 输入净化：20+ 中英文注入模式匹配，覆盖常见提示词注入
- 输出脱敏：手机号、身份证号、银行卡号的正则匹配和替换
- 审计日志：每次查询记录用户ID、查询内容、工具调用链、响应耗时

### 工具注册中心

统一的装饰器模式工具注册接口，自动收集工具元数据供Agent调用：
- search_policy(query) - 制度条款语义检索
- calculate_loan(principal, annual_rate, years) - 等额本息贷款计算

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| DEEPSEEK_API_KEY | (内置) | Deepseek API密钥 |
| DEEPSEEK_API_BASE | https://api.deepseek.com | API基础地址 |
| DEEPSEEK_MODEL | deepseek-v4-flash | 模型名称 |
| KNOWLEDGE_BASE_PATH | data/knowledge_base.json | 知识库路径 |
| RETRIEVAL_TOP_K | 3 | 检索返回数量 |
| RETRIEVAL_SIMILARITY_THRESHOLD | 0.1 | 相似度阈值 |
| HOST | 0.0.0.0 | 服务监听地址 |
| PORT | 8000 | 服务端口 |
| DEBUG | false | 调试模式 |
| AUDIT_LOG_PATH | logs/audit.log | 审计日志路径 |

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 编程语言 | Python >= 3.10 |
| Web框架 | FastAPI >= 0.115 |
| LLM | Deepseek-v4-flash |
| Agent框架 | 原生Python（ReAct模式） |
| 知识库 | JSON文件 + TF-IDF检索 |
| 语义检索 | TF-IDF + 余弦相似度 |
| 输入净化 | 正则匹配 |
| 审计日志 | Python logging |
| 测试框架 | pytest |
