"""输出脱敏模块.

对敏感个人信息（手机号、身份证号、银行卡号）进行正则匹配和脱敏处理。
"""

from __future__ import annotations

import re
from typing import ClassVar


class OutputDesensitizer:
    """输出脱敏器.

    对文本中的个人信息数据进行脱敏，支持手机号、身份证号、银行卡号。

    Examples:
        >>> d = OutputDesensitizer()
        >>> d.desensitize("手机号13800138000")
        '手机号138****8000'
        >>> d.desensitize("身份证110101199001011234")
        '身份证110101********1234'
        >>> d.desensitize("银行卡6222021234567890")
        '银行卡622202******7890'
    """

    # 手机号：11位数字，以1开头
    PHONE_PATTERN: ClassVar[str] = r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)"
    PHONE_REPLACEMENT: ClassVar[str] = r"\1****\2"

    # 身份证号：18位（最后一位可能为X）
    ID_CARD_PATTERN: ClassVar[str] = r"(?<!\d)(\d{6})\d{8}(\d{4})(?!\d)"
    ID_CARD_REPLACEMENT: ClassVar[str] = r"\1********\2"

    # 银行卡号：16-19位数字
    BANK_CARD_PATTERN: ClassVar[str] = r"(?<!\d)(\d{6})\d{6,10}(\d{4})(?!\d)"
    BANK_CARD_REPLACEMENT: ClassVar[str] = r"\1******\2"

    def __init__(self) -> None:
        """初始化输出脱敏器，编译正则表达式。"""
        self._phone_re = re.compile(self.PHONE_PATTERN)
        self._id_card_re = re.compile(self.ID_CARD_PATTERN)
        self._bank_card_re = re.compile(self.BANK_CARD_PATTERN)

    def desensitize(self, text: str) -> str:
        """对文本中的敏感信息进行脱敏处理.

        Args:
            text: 原始文本

        Returns:
            脱敏后的文本
        """
        result = self._phone_re.sub(self.PHONE_REPLACEMENT, text)
        result = self._id_card_re.sub(self.ID_CARD_REPLACEMENT, result)
        result = self._bank_card_re.sub(self.BANK_CARD_REPLACEMENT, result)
        return result


# 全局单例
output_desensitizer = OutputDesensitizer()
