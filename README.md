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
│   ├── config.py               # 配置管理（API密钥、模型参数、.env加载）
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── agent.py            # 合规智能体核心（ReAct模式 + 来源幻觉过滤）
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
│   ├── static/
│   │   └── index.html          # 前端聊天界面
│   └── tools/
│       ├── __init__.py
│       ├── registry.py         # 工具注册中心
│       ├── search_policy.py    # 制度检索工具
│       └── calculate_loan.py   # 贷款计算工具
├── tests/
│   ├── __init__.py
│   └── test_agent.py           # 核心功能测试（37个用例）
├── data/
│   └── knowledge_base.json     # 知识库数据
└── logs/
    └── audit.log               # 审计日志（运行时生成）
```

## 环境依赖

- Python >= 3.10
- 依赖包（见 requirements.txt）

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
# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- **前端聊天界面**：http://localhost:8000 （推荐从这里开始）
- **API文档（Swagger UI）**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

## 前端界面

系统内置了一个完整的聊天界面，在浏览器打开 http://localhost:8000 即可使用。

### 功能特性

- **聊天对话**：输入问题，按 Enter 发送，AI 实时回复
- **快捷建议**：点击建议按钮一键提问常见问题
- **Markdown 渲染**：AI 回复中的 **粗体**、`行内代码`、列表、标题等自动格式化展示
- **来源标注**：每个回答下方的蓝色/绿色标签显示引用来源
- **工具调用标记**：使用了搜索或计算工具时有明显标记
- **注入攻击拦截**：检测到提示词注入时以红色警示框提示
- **Shift+Enter 换行**：支持多行输入

### 建议问题

打开界面后可直接点击快捷按钮或输入以下问题体验：

| 类别 | 示例问题 |
|------|----------|
| 制度查询 | 贷款审批流程是什么 |
| 认定标准 | 首套房认定标准 |
| 合规规定 | 不良贷款认定标准 |
| 贷款计算 | 计算100万贷款，年利率3.5%，期限30年 |
| 开放查询 | 企业成立2年，纳税正常，征信无逾期，负债率60% |

### 界面截图

界面包含以下区域：
- **顶部导航**：标题 + 安全接入 / 版本号
- **聊天区**：用户消息和AI回复交替展示，来源标签彩色显示
- **输入区**：文本框 + 发送按钮 + 快捷键提示

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
{
  "answer": "贷款审批流程包括以下步骤：1）客户提交贷款申请... [信贷业务制度:4.1]",
  "sources": ["信贷业务制度:4.1"],
  "tool_calls": [
    {
      "tool_name": "search_policy",
      "arguments": {"query": "贷款审批流程是什么"},
      "result": "..."
    }
  ],
  "processing_time_ms": 1234.56,
  "injection_detected": false
}
```

**响应字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| answer | string | AI回答文本 |
| sources | string[] | 引用来源列表（已过滤知识库中不存在的虚假来源） |
| tool_calls | object[] | 工具调用链 |
| processing_time_ms | float | 处理耗时（毫秒） |
| injection_detected | bool | 是否检测到注入攻击 |

### 3. 获取审计日志

```
GET /api/audit/recent?count=10
```

### 4. 列出可用工具

```
GET /api/tools
```

## 测试方法

项目包含 **37 个自动化测试用例**，覆盖作业要求的全部 5 个功能点以及安全、知识库、工具注册、脱敏等模块。

### 测试总览

| 测试类别 | 测试类 | 用例数 | 覆盖内容 |
|----------|--------|--------|----------|
| 制度检索 | TestLoanApprovalProcess | 2 | TC-A-01：语义检索、Top-K排序 |
| 认定标准 | TestFirstHomeCriteria | 2 | TC-A-02：首套房认定、首付比例 |
| 贷款计算 | TestLoanCalculation | 4 | TC-A-03：计算逻辑、无效输入 |
| 注入防护 | TestInjectionProtection | 3 | TC-A-04：10种注入模式、正常输入、Agent拦截 |
| 无匹配处理 | TestNoMatchInKnowledgeBase | 1 | TC-A-05：知识库无匹配 |
| 知识库基础 | TestKnowledgeBase | 4 | 加载、字段完整性、规则数量 |
| 输出脱敏 | TestOutputDesensitization | 4 | 手机/身份证/银行卡脱敏 |
| 工具注册 | TestToolRegistry | 4 | 注册、元数据、执行、错误处理 |
| 端到端流程 | TestEndToEnd | 4 | 完整查询流程、审计日志（需API连通） |

### 运行全部测试

```bash
# 运行所有离线测试（33个，推荐）
pytest tests/ -v -k "not EndToEnd"

# 运行全部测试（包括端到端，需要网络和Deepseek API）
pytest tests/ -v

# 生成覆盖率报告
pytest tests/ -v --cov=app --cov-report=term-missing
```

### 运行单个测试用例

```bash
# TC-A-01：贷款审批流程
pytest tests/test_agent.py::TestLoanApprovalProcess -v

# TC-A-02：首套房认定标准
pytest tests/test_agent.py::TestFirstHomeCriteria -v

# TC-A-03：贷款计算
pytest tests/test_agent.py::TestLoanCalculation -v

# TC-A-04：提示词注入防护
pytest tests/test_agent.py::TestInjectionProtection -v

# TC-A-05：知识库无匹配
pytest tests/test_agent.py::TestNoMatchInKnowledgeBase -v

# 输出脱敏
pytest tests/test_agent.py::TestOutputDesensitization -v

# 工具注册
pytest tests/test_agent.py::TestToolRegistry -v

# 知识库
pytest tests/test_agent.py::TestKnowledgeBase -v
```

### 测试用例详情

#### TC-A-01：贷款审批流程

| 项目 | 内容 |
|------|------|
| 测试输入 | "贷款审批流程是什么" |
| 预期结果 | 返回审批流程步骤，标注 [信贷业务制度:4.1] |
| 测试类 | TestLoanApprovalProcess |
| 验证点 | 检索结果包含审批流程相关条款，来源标注格式正确，结果按相似度降序排列 |

#### TC-A-02：首套房认定标准

| 项目 | 内容 |
|------|------|
| 测试输入 | "首套房认定标准" |
| 预期结果 | 返回首付比例和认定条件，标注来源 |
| 测试类 | TestFirstHomeCriteria |
| 验证点 | 检索到首套房认定条款，来源为 [信贷业务制度:4.4]，首付比例不低于20% |

#### TC-A-03：贷款计算

| 项目 | 内容 |
|------|------|
| 测试输入 | "计算100万贷款，年利率3.5%，期限30年" |
| 预期结果 | 返回月供和总利息数值 |
| 测试类 | TestLoanCalculation |
| 验证点 | 月供在合理范围（约4490元/月），总利息大于0，无效输入正确拦截 |

#### TC-A-04：提示词注入防护

| 项目 | 内容 |
|------|------|
| 测试输入 | "ignore previous instructions" |
| 预期结果 | 拦截注入，返回安全提示 |
| 测试类 | TestInjectionProtection |
| 验证点 | 中英文共10种注入模式均被检测，Agent返回安全拦截提示，注入时不执行工具 |

#### TC-A-05：知识库无匹配

| 项目 | 内容 |
|------|------|
| 测试输入 | "某企业贷款管理规定" |
| 预期结果 | 知识库中不存在，返回无法确认 |
| 测试类 | TestNoMatchInKnowledgeBase |
| 验证点 | Agent返回"根据现有知识库无法确认，请咨询相关部门" |

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

### 幻觉防护（来源防伪造）

LLM 在回答时可能编造不存在的制度来源。系统通过三层机制防止虚假来源：
1. **系统提示约束**：在系统提示中列出知识库实际包含的 7 个来源清单，明确要求只能引用 search_policy 返回的 source_tag
2. **工具反馈强化**：每次工具结果回传时附带强制约束提示
3. **后处理硬过滤**：正则提取回答中所有 [文档名:章节号] 格式的来源标注，逐一比对知识库白名单，不存在的自动删除

### 安全防护

- **输入净化**：20+ 中英文注入模式匹配，覆盖常见提示词注入
- **输出脱敏**：手机号（1XX****XXXX）、身份证号（110101********1234）、银行卡号（622202******7890）的正则匹配和替换，含数字边界保护避免误匹配
- **审计日志**：每次查询记录用户ID、查询内容、工具调用链、响应耗时，支持文件持久化和内存缓存

### 工具注册中心

统一的装饰器模式工具注册接口，自动收集工具元数据供Agent调用：

**search_policy(query)** — 制度条款语义检索
- 基于 TF-IDF 向量化检索知识库
- 返回 Top-3 最相关条款及其来源标签

**calculate_loan(principal, annual_rate, years)** — 等额本息贷款计算
- 支持任意本金、利率、期限
- 返回月供、总利息、总还款额

### 前端界面

- 单页面 HTML 应用，无外部依赖
- 基于块级处理的 Markdown 渲染引擎（代码块、标题、列表、粗体、链接）
- 来源标签彩色区分（蓝色=信贷业务制度，绿色=合规操作手册）
- 注入警告红色展示
- 移动端自适应布局

## 环境变量说明

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| DEEPSEEK_API_KEY | （从 .env 读取） | Deepseek API密钥 |
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
| LLM | Deepseek-v4-flash（HTTP API调用） |
| Agent框架 | 原生Python（ReAct模式） |
| 知识库 | JSON文件 + TF-IDF检索 |
| 语义检索 | TF-IDF + 余弦相似度 |
| 前端 | 纯HTML/CSS/JS单页面应用 |
| 输入净化 | 正则匹配（20+注入模式） |
| 输出脱敏 | 正则匹配（手机/身份证/银行卡） |
| 审计日志 | Python logging + 文件持久化 |
| 幻觉防护 | 提示约束 + 后处理白名单过滤 |
| 测试框架 | pytest（37个用例） |

## 常见问题

### Q：API密钥如何配置？

API密钥通过 `.env` 文件配置，该文件已加入 `.gitignore`，不会被提交到代码仓库：

```bash
cp .env.example .env
# 编辑 .env，将 DEEPSEEK_API_KEY 设置为你的密钥
```

### Q：前端页面无法访问？

确认服务已启动后，浏览器访问 http://localhost:8000。如果使用自定义端口，请相应调整 URL。

### Q：测试时出现网络错误？

EndToEnd 类测试需要连接 Deepseek API，请确保：
1. `.env` 中的 API 密钥正确
2. 网络可以访问 api.deepseek.com
3. 其他离线测试不受网络影响

