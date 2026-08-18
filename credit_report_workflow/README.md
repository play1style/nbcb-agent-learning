# 主题2：对公信贷尽调报告生成工作流（Python 版）

## 运行方式

### 基础运行（Mock 模式，无需 API Key）

```bash
cd python
python main.py
```

### 真实 LLM 增强模式

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

## 文件说明

- `main.py`：工作流入口
- `input_step.py`：输入企业数据
- `basic_info_step.py`：整理基本信息
- `financial_analysis_step.py`：财务指标计算
- `risk_assessment_step.py`：风险判断
- `risk_rules.json`：风险规则配置，可调整阈值与提示文案
- `report_generation_step.py`：生成结构化报告（含 LLM 增强 + Mock 降级）
- `llm_client.py`：LLM 调用封装（标准库实现，零依赖）

## 环境要求

- Python 3.7+
- 核心逻辑无第三方依赖，仅使用 Python 标准库
- 真实 LLM 模式需要网络环境和有效 API Key

## 设计说明

本案例在保持前 4 个工作流节点（输入 → 基本信息 → 财务分析 → 风险评估）不变的前提下，
于第 5 步"报告生成"中引入了**大模型增强**能力：

- **有 API Key 时**：调用 LLM 根据企业数据和风险点生成专业、自然的初步结论
- **无 API Key 或调用失败时**：自动降级为模板拼接，确保代码始终可运行

这体现了银行场景中"可控生成"的典型实践：**模板保底线 + LLM 提质量**。
