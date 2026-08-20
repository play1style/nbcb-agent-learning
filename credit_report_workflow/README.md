# 主题2：对公信贷尽调报告生成工作流（Python 版）

基于 5 个可编排节点，将企业原始数据加工为一份结构化授信尽调报告初稿。
报告结论支持大模型增强生成，无 API Key 或调用失败时自动降级为模板文案，保证流程始终可运行。

## 工作流

```text
步骤1 输入企业数据       input_step.py
步骤2 整理基本信息       basic_info_step.py
步骤3 财务指标分析       financial_analysis_step.py
步骤4 风险规则判断       risk_assessment_step.py
步骤5 生成结构化报告     report_generation_step.py
步骤6 导出报告           report_exporter.py -> export/*.txt
```

各步骤通过同一个 `state` 字典逐级传递数据：原始输入 → 基本信息 → 财务指标 → 风险列表 → 报告文本。

## 快速开始

### 1. 安装依赖

- Python 3.7+
- `python-dotenv`（用于读取 `.env` 配置）

```bash
pip install -r requirements.txt
```

### 2. Mock 模式（无需 API Key）

```bash
python main.py
```

未检测到 `LLM_API_KEY` 时，程序自动进入 Mock 模式，初步结论使用固定模板文案。

### 3. LLM 增强模式

```bash
# Linux / macOS
export LLM_API_KEY="你的API密钥"
export LLM_BASE_URL="https://api.moonshot.cn/v1"  # 默认，可选
export LLM_MODEL="moonshot-v1-8k"                  # 默认，可选
python main.py

# Windows CMD
set LLM_API_KEY=你的API密钥
python main.py

# Windows PowerShell
$env:LLM_API_KEY="你的API密钥"
python main.py
```

也可以创建 `.env` 文件（已加入 `.gitignore`，不会提交到仓库）：

```bash
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://api.moonshot.cn/v1
LLM_MODEL=moonshot-v1-8k
LLM_TIMEOUT=10
LLM_MOCK=false
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | API 密钥；未设置时自动进入 Mock 模式 | 空 |
| `LLM_BASE_URL` | OpenAI 兼容接口基地址 | `https://api.moonshot.cn/v1` |
| `LLM_MODEL` | 模型名称 | `moonshot-v1-8k` |
| `LLM_TIMEOUT` | 请求超时秒数 | `10` |
| `LLM_MOCK` | 模式控制：`auto`（默认）\| `true` \| `false` | `auto` |

`LLM_MOCK=true` 强制 Mock，`LLM_MOCK=false` 强制调用 LLM，`auto` 按是否配置 API Key 自动判断。

## 配置说明

### 企业数据

修改 `input_step.py` 中 `get_input()` 返回的字典，即可更换企业名称、行业、营业收入、净利润、应收账款周转天数、总资产、总负债等原始数据。

### 风险规则

风险判断逻辑读取 `risk_rules.json`，调整阈值或提示文案无需修改代码：

- `source`：`input` 表示取原始输入字段，`financial` 表示取财务指标
- `operator`：`gt` / `gte` / `lt` / `lte`
- `value`：触发阈值
- `message`：风险提示文案
- `no_risk_message`：无风险时的兜底文案

## 报告结构

输出报告包含五个部分：

1. 企业基本信息
2. 财务指标分析（销售净利率、资产负债率、应收账款周转次数）
3. 风险判断
4. 行业提示
5. 初步结论（LLM 生成或模板兜底）

报告以 `yyyyMMddHHMMSS + 两位随机数.txt` 的格式写入 `export/` 目录。

## 设计说明

本案例在保持前 4 个节点不变的前提下，于第 5 步引入大模型增强：

- 有 API Key 时：调用 LLM 基于企业数据与风险点生成专业、自然的初步结论
- 无 API Key 或调用失败时：自动降级为模板拼接，确保代码始终可运行

体现银行场景中"可控生成"的典型实践：**模板保底线 + LLM 提质量**。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 工作流入口，串联 5 个处理步骤并导出报告 |
| `input_step.py` | 步骤1：输入企业原始数据 |
| `basic_info_step.py` | 步骤2：整理企业基本信息 |
| `financial_analysis_step.py` | 步骤3：计算财务指标 |
| `risk_assessment_step.py` | 步骤4：加载规则并判断风险 |
| `risk_rules.json` | 风险规则配置 |
| `report_generation_step.py` | 步骤5：生成结构化报告（LLM 增强 + Mock 降级） |
| `llm_client.py` | LLM 调用封装，支持 `.env` 与环境变量配置 |
| `report_exporter.py` | 将报告导出为 txt 文件 |
| `export/` | 生成的报告输出目录 |
| `images/` | Mock / LLM 两种模式的运行截图 |
