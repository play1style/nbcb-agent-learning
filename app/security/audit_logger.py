"""审计日志模块.

记录每次查询的用户ID、查询内容、工具调用链、响应耗时等信息。
支持文件日志和内存缓存两种模式。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import config

# 确保日志目录存在
_log_path = Path(config.AUDIT_LOG_PATH)
_log_path.parent.mkdir(parents=True, exist_ok=True)

# 配置审计日志记录器
_logger = logging.getLogger("audit")
_logger.setLevel(logging.INFO)
_handler = logging.FileHandler(str(_log_path), encoding="utf-8")
_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
_logger.addHandler(_handler)


class AuditLogger:
    """审计日志记录器.

    记录每次查询的完整审计信息，支持文件持久化和内存缓存。

    Examples:
        >>> logger = AuditLogger()
        >>> logger.log(
        ...     user_id="user001",
        ...     query="审批流程",
        ...     tool_chain=[{"tool": "search_policy", "args": {"query": "审批流程"}}],
        ...     response_time_ms=123.4,
        ...     injection_detected=False,
        ...     answer_preview="贷款审批流程包括..."
        ... )
    """

    def __init__(self) -> None:
        """初始化审计日志记录器。"""
        self._recent_logs: list[dict[str, Any]] = []
        self._max_cache: int = 100

    def log(
        self,
        user_id: str,
        query: str,
        tool_chain: list[dict[str, Any]],
        response_time_ms: float,
        injection_detected: bool = False,
        answer_preview: str = "",
    ) -> None:
        """记录一条审计日志.

        Args:
            user_id: 用户标识
            query: 查询内容
            tool_chain: 工具调用链
            response_time_ms: 响应耗时（毫秒）
            injection_detected: 是否检测到注入攻击
            answer_preview: 回答摘要
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "query": query[:200],  # 截断过长查询
            "tool_chain": tool_chain,
            "response_time_ms": round(response_time_ms, 2),
            "injection_detected": injection_detected,
            "answer_preview": answer_preview[:200],
        }

        # 写入文件
        _logger.info(json.dumps(entry, ensure_ascii=False))

        # 内存缓存
        self._recent_logs.append(entry)
        if len(self._recent_logs) > self._max_cache:
            self._recent_logs.pop(0)

    def get_recent_logs(self, count: int = 10) -> list[dict[str, Any]]:
        """获取最近的审计日志.

        Args:
            count: 返回条目数

        Returns:
            最近的日志条目列表
        """
        return self._recent_logs[-count:]


# 全局单例
audit_logger = AuditLogger()
