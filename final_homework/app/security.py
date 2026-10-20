"""提示词注入过滤和敏感信息脱敏。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


INJECTION_PATTERNS = [
    # 同时覆盖典型英文越狱语句和中文提示词覆盖语句。
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"reveal\s+(the\s+)?(system|developer)\s+(prompt|instructions?)",
        r"show\s+(me\s+)?(the\s+)?(system|developer)\s+(prompt|instructions?)",
        r"忽略(?:之前|先前|以上|前面)的?(?:所有)?(?:指令|指示|要求)",
        r"无视(?:之前|先前|以上|前面)的?(?:所有)?(?:指令|指示|要求)",
        r"(?:泄露|展示|输出|打印).{0,8}(?:系统|开发者).{0,6}(?:提示词|指令)",
        r"jailbreak|developer\s+mode|role\s*override",
    )
]

MOBILE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
ID_CARD_PATTERN = re.compile(
    r"(?<!\d)([1-9]\d{5}(?:18|19|20)\d{2}"
    r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"
)
BANK_CARD_PATTERN = re.compile(
    r"(?<!\d)(\d{4})[ -]?(\d{4})[ -]?(\d{4})[ -]?(\d{4,7})(?!\d)"
)
EMAIL_PATTERN = re.compile(
    r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
)


@dataclass(slots=True)
class SanitizationResult:
    """用户输入净化与安全检测结果。"""

    text: str
    blocked: bool
    reason: str = ""


def sanitize_input(text: str, max_length: int = 2000) -> SanitizationResult:
    """规范化用户输入并检测显式提示词注入。"""
    # NFKC 可统一全角、半角及部分兼容字符，降低绕过风险。
    normalized = unicodedata.normalize("NFKC", text or "")
    # 移除非必要的控制字符，仅保留换行和制表符。
    normalized = "".join(
        char
        for char in normalized
        if char in {"\n", "\t"} or unicodedata.category(char)[0] != "C"
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    if len(normalized) > max_length:
        normalized = normalized[:max_length].rstrip()
    for pattern in INJECTION_PATTERNS:
        if pattern.search(normalized):
            return SanitizationResult(
                text=normalized,
                blocked=True,
                reason="检测到提示词注入或角色覆盖意图",
            )
    return SanitizationResult(text=normalized, blocked=False)


def redact_text(text: str) -> str:
    """脱敏手机号、身份证号、银行卡号和邮箱地址。"""
    if not text:
        return text

    def mask_mobile(match: re.Match) -> str:
        """保留手机号前三位和后四位。"""
        value = match.group(1)
        return f"{value[:3]}****{value[-4:]}"

    def mask_id_card(match: re.Match) -> str:
        """保留身份证前六位和后四位。"""
        value = match.group(1)
        return f"{value[:6]}********{value[-4:]}"

    def mask_bank_card(match: re.Match) -> str:
        """保留银行卡前四位和后四位。"""
        parts = match.groups()
        digits = "".join(parts)
        if len(digits) < 12:
            return match.group(0)
        return f"{digits[:4]} **** **** {digits[-4:]}"

    def mask_email(match: re.Match) -> str:
        """仅保留邮箱首字符和域名。"""
        local, domain = match.groups()
        visible = local[:1] if local else "*"
        return f"{visible}***@{domain}"

    # 先处理身份证，避免 18 位号码被银行卡规则优先截断。
    redacted = ID_CARD_PATTERN.sub(mask_id_card, text)
    redacted = MOBILE_PATTERN.sub(mask_mobile, redacted)
    redacted = BANK_CARD_PATTERN.sub(mask_bank_card, redacted)
    redacted = EMAIL_PATTERN.sub(mask_email, redacted)
    return redacted


def injection_response() -> str:
    """返回提示词注入被拦截时的标准安全提示。"""
    return (
        "检测到可能的提示词注入，已拦截该请求。"
        "请仅提交与银行制度查询或贷款计算相关的问题。"
    )
