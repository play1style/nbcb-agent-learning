"""安全防护模块——输入净化（拦截提示词注入）与输出脱敏（隐藏敏感信息）。"""

import re
from typing import List, Tuple

# ╔══════════════════════════════════════════════════════════════╗
# ║  提示词注入攻击检测模式库                                    ║
# ╚══════════════════════════════════════════════════════════════╝

_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    # 英文注入模式
    (r"(?i)ignore\s+(previous|above|all)\s+(instructions|prompts|context|directions)", "INJ-01"),
    (r"(?i)forget\s+(everything|all|previous|your)\s", "INJ-02"),
    (r"(?i)you\s+(are|were)\s+(now|not)\s", "INJ-03"),
    (r"(?i)(system|new)\s+(prompt|instruction|message)", "INJ-04"),
    (r"(?i)override\s+(instructions|system|rules|commands)", "INJ-05"),
    # 中文注入模式
    (r"(?i)(你(现在|必须|要)(是|成为|扮演)|请你扮演)", "INJ-06"),
    (r"(?i)忽略\s*(前面|以上|所有)\s*(指令|指示|要求|的内容)", "INJ-07"),
    (r"(?i)请(忘记|忽略)\s*(一切|所有|之前)", "INJ-08"),
    (r"(?i)不要(遵守|遵循|按照)(前面|以上)", "INJ-09"),
    (r"(?i)(请|现在)重新(设置|设定|定义|配置)", "INJ-10"),
]


def sanitize_input(text: str) -> Tuple[str, bool]:
    """
    检测并拦截提示词注入攻击。

    Args:
        text: 用户原始输入。

    Returns:
        返回 (安全文本 或 "[BLOCKED]", 是否被拦截)。
    """
    text_stripped = text.strip()
    for pattern, code in _INJECTION_PATTERNS:
        if re.search(pattern, text_stripped):
            return "[BLOCKED]", True
    return text, False


def desensitize(text: str) -> str:
    """
    对输出文本中的敏感信息进行脱敏处理。

    处理范围：
    - 手机号（1[3-9] 开头，11 位）
    - 身份证号（18 位）
    - 银行卡号（16-19 位）

    Args:
        text: 待脱敏的原始文本。

    Returns:
        脱敏后的安全文本。
    """
    # 手机号脱敏：保留前 3 位和末 4 位，中间掩码
    text = re.sub(
        r'(1[3-9]\d)\s*-?\s*(\d{4})\s*-?\s*(\d{4})',
        r'\1****\3',
        text,
    )
    # 身份证号脱敏：保留前 6 位和末 4 位，中间 8 位掩码
    text = re.sub(
        r'\b(\d{6})\s*\d{8}\s*(\d{4})\b',
        r'\1********\2',
        text,
    )
    # 银行卡号脱敏：保留前 4 位和末 4 位，中间掩码
    text = re.sub(
        r'\b(\d{4})\s*\d{8,11}\s*(\d{4})\b',
        r'\1********\2',
        text,
    )
    return text
