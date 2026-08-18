"""
步骤5：根据结构化模板生成尽调报告。

可控文本生成的核心体现：
- 报告整体框架由模板固定（格式可控）
- 财务数据和风险点由规则计算得出（内容可控）
- 初步结论段落优先调用大模型生成（智能增强），
  无 API Key 或调用失败时自动降级为模板拼接（可靠兜底）
"""

from llm_client import LLMClient


SYSTEM_PROMPT = (
    "你是一位资深的银行对公信贷客户经理，擅长撰写授信尽调报告。"
    "请根据提供的企业信息和财务指标，生成一段专业、严谨的初步结论，"
    "控制在 120 字以内。必须基于真实数据，不得虚构。"
)


def generate_report(state):
    """按模板生成结构化尽调报告。"""
    # 1. 先拼接除结论外的固定部分
    basic = format_section(state["basic_info"])
    financial = format_section(state["financial"])
    risks = "\n".join(f"- {r}" for r in state["risks"])

    # 2. 尝试调用 LLM 生成初步结论
    conclusion = _generate_conclusion_by_llm(state)
    if conclusion is None:
        # Mock 降级：使用模板结论
        conclusion = "建议结合行业周期及客户上下游账期进一步核实，审慎核定授信额度。"

    report = f"""尽调报告初稿
============

一、企业基本信息
{basic}

二、财务指标分析
{financial}

三、风险判断
{risks}

四、行业提示
该企业所属纺织服装行业，应收账款周期普遍较长，建议重点关注下游客户回款情况及存货周转效率。

五、初步结论
{conclusion}
"""
    return report


def _generate_conclusion_by_llm(state):
    """
    调用 LLM 生成初步结论段落。
    若 LLM 不可用则返回 None，由外层降级为模板结论。
    """
    client = LLMClient()

    # 将 state 组织成简洁的提示词
    info = state["basic_info"]
    fin = state["financial"]
    risks = state["risks"]

    user_prompt = f"""请为以下企业撰写尽调报告的初步结论段落。

企业基本信息：
{format_prompt_dict(info)}

财务指标：
{format_prompt_dict(fin)}

风险点：
{chr(10).join(risks)}

要求：
- 语言专业、简洁，体现银行信贷视角
- 必须包含风险提示和授信建议倾向
- 不要编造数据中未提及的内容
"""

    return client.chat(SYSTEM_PROMPT, user_prompt, max_tokens=256)


def format_section(data):
    """将字典格式化为 key: value 的文本块。"""
    return "\n".join(f"{k}: {v}" for k, v in data.items())


def format_prompt_dict(data):
    """将字典格式化为用于 Prompt 的文本。"""
    return "\n".join(f"- {k}: {v}" for k, v in data.items())
