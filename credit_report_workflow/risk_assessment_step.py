"""
步骤4：根据财务指标和业务规则判断风险点。

风险规则从 risk_rules.json 配置文件中读取，
调整阈值或提示文案时无需修改代码。
"""

import json
from pathlib import Path


CONFIG_FILE = Path(__file__).resolve().parent / "risk_rules.json"

OPERATORS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}


def load_rules():
    """读取风险规则配置。"""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        config = json.load(f)
    return config["rules"], config["no_risk_message"]


def _get_metric(state, rule):
    """从原始输入或财务指标中取出规则需要比较的数值。"""
    if rule["source"] == "input":
        return state[rule["metric"]]
    if rule["source"] == "financial":
        return state["financial"][rule["metric"]]
    raise ValueError(f"未知的规则数据来源: {rule.get('source')}")


def assess_risk(state):
    """基于配置文件中的规则判断风险。"""
    rules, no_risk_message = load_rules()
    risks = []

    for rule in rules:
        metric = _get_metric(state, rule)
        if OPERATORS[rule["operator"]](metric, rule["value"]):
            risks.append(rule["message"])

    if not risks:
        risks.append(no_risk_message)

    state["risks"] = risks
    return state
