"""为每次合规查询写入结构化审计日志。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class AuditRecord:
    """一次完整查询的审计事件。"""

    request_id: str
    user_id: str
    session_id: str
    query: str
    query_hash: str
    tool_chain: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: str = "低"
    confidence_score: float = 0.0
    blocked: bool = False
    latency_ms: int = 0
    response_hash: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class AuditLogger:
    """将审计事件追加写入统一 JSONL 日志流。"""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / "audit.jsonl"

    def record(
        self,
        record: AuditRecord,
        response: str,
    ) -> None:
        """持久化审计事件，并计算响应内容完整性哈希。"""
        # 响应哈希方便后续确认审计日志中的回答是否被篡改。
        record.response_hash = hashlib.sha256(
            response.encode("utf-8")
        ).hexdigest()
        lock_path = self.path.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            # 多进程并发写入时使用独占锁，避免 JSON 行相互覆盖。
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(asdict(record), ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    # 审计记录必须尽快落盘，不应只停留在进程缓冲区。
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def read_recent(
        self,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """读取最近审计事件，供管理接口展示。"""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user_id and item.get("user_id") != user_id:
                continue
            records.append(item)
        return records[-max(1, limit) :]
