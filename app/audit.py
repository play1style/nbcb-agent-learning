"""审计日志模块——每次查询都会以 JSON Lines 格式记录到文件。"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditLogger:
    """审计日志记录器，将每一次人机交互持久化为 JSON Lines 记录。"""

    def __init__(self, log_file: Optional[str] = None):
        """
        初始化审计日志记录器。

        Args:
            log_file: 审计日志文件路径。不传则使用默认路径。
        """
        self.log_file = log_file or str(Path(__file__).parent.parent / "logs" / "audit.log")
        self._ensure_log_file()
        self.logger = logging.getLogger("audit")
        self._setup_logger()

    # ── 内部辅助方法 ─────────────────────────────────────────────────

    def _ensure_log_file(self) -> None:
        """确保日志文件所在目录存在。"""
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> None:
        """配置 FileHandler，将所有日志写入 JSON Lines 文件。"""
        handler = logging.FileHandler(self.log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    # ── 公开接口 ─────────────────────────────────────────────────────

    def log(
        self,
        user_id: str,
        query: str,
        sanitized_query: str,
        tool_calls: List[Dict[str, Any]],
        response: str,
        processing_time_ms: int,
    ) -> None:
        """
        持久化一次人机交互记录。

        Args:
            user_id:           用户标识。
            query:             原始查询（未经净化）。
            sanitized_query:   净化后的查询。
            tool_calls:        工具调用序列。
            response:          助手的最终回答。
            processing_time_ms: 处理耗时（毫秒）。
        """
        entry = {
            "timestamp":          datetime.now().isoformat(),
            "user_id":            user_id,
            "query":              query,
            "sanitized_query":    sanitized_query,
            "tool_calls":         tool_calls,
            "response":           response,
            "processing_time_ms": processing_time_ms,
        }
        self.logger.info(json.dumps(entry, ensure_ascii=False))

    def get_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        返回最近的 *limit* 条审计记录。

        Args:
            limit: 最多返回的条目数。
        """
        logs: List[Dict[str, Any]] = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            return []
        return logs[-limit:]
