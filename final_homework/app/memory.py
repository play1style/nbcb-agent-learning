"""文件持久化会话记忆和带哈希链的合规备份。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import MemoryMessage


def _utc_now() -> str:
    """返回 ISO-8601 格式的 UTC 时间戳。"""
    return datetime.now(timezone.utc).isoformat()


class ConversationMemory:
    """按会话追加写入 JSONL，并渲染后续轮次使用的记忆上下文。"""

    def __init__(self, data_dir: Path, window_turns: int = 6) -> None:
        self.root = data_dir / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)
        self.window_turns = window_turns

    def load_messages(
        self,
        session_id: str,
        user_id: str | None = None,
        limit_turns: int | None = None,
    ) -> list[MemoryMessage]:
        """读取指定用户和会话的历史消息。"""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        messages: list[MemoryMessage] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if user_id and item.get("user_id") != user_id:
                    continue
                messages.append(MemoryMessage.model_validate(item))
        except (json.JSONDecodeError, ValueError):
            return []
        # 每轮包含一条用户消息和一条助手消息。
        max_messages = (limit_turns or self.window_turns) * 2
        return messages[-max_messages:]

    def append_turn(
        self,
        session_id: str,
        user_message: MemoryMessage,
        assistant_message: MemoryMessage,
    ) -> None:
        """在跨进程文件锁保护下追加一轮用户和助手消息。"""
        path = self._session_path(session_id)
        with self._locked(path) as handle:
            handle.seek(0, os.SEEK_END)
            for message in (user_message, assistant_message):
                handle.write(
                    json.dumps(
                        message.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            # 强制刷新到磁盘，降低进程异常退出导致记录丢失的风险。
            os.fsync(handle.fileno())

    def render_context(
        self,
        messages: list[MemoryMessage],
        max_chars: int = 4000,
    ) -> str:
        """渲染最近会话内容，供每轮提示词自动拼接。"""
        lines = []
        for message in messages[-self.window_turns * 2 :]:
            role = "用户" if message.role == "user" else "助手"
            lines.append(f"{role}：{message.content}")
        text = "\n".join(lines)
        return text[-max_chars:]

    def _session_path(self, session_id: str) -> Path:
        """生成安全且受限于白名单字符的会话文件路径。"""
        if not session_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for char in session_id
        ):
            raise ValueError("非法 session_id")
        return self.root / f"{session_id}.jsonl"

    @contextmanager
    def _locked(self, path: Path) -> Iterator[Any]:
        """使用独占咨询锁打开会话文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            # 多 worker 同时写入时，通过 flock 保证记录不会交错。
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                with path.open("a+", encoding="utf-8") as handle:
                    yield handle
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


class ComplianceBackupStore:
    """按日期写入带哈希链的会话备份，供合规检查和完整性验证。"""

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "compliance"
        self.root.mkdir(parents=True, exist_ok=True)

    def append(self, session_id: str, record: dict[str, Any]) -> str:
        """追加一条备份记录并返回其完整性哈希。"""
        day = date.today().isoformat()
        path = self.root / day / "conversations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                previous_hash = self._last_hash(path)
                # 每条记录引用上一条哈希，形成可持续校验的链。
                payload = {
                    **record,
                    "session_id": session_id,
                    "backup_time": _utc_now(),
                    "previous_hash": previous_hash,
                }
                record_hash = self._hash_record(payload)
                payload["record_hash"] = record_hash
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(payload, ensure_ascii=False) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                return record_hash
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _last_hash(path: Path) -> str:
        """读取备份文件中最后一条记录的哈希。"""
        if not path.exists():
            return "GENESIS"
        last = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last = line
        if not last:
            return "GENESIS"
        try:
            return json.loads(last).get("record_hash", "GENESIS")
        except json.JSONDecodeError:
            return "GENESIS"

    @staticmethod
    def _hash_record(payload: dict[str, Any]) -> str:
        """使用规范化 JSON 生成确定性 SHA-256 哈希。"""
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AtomicJsonWriter:
    """用于可靠写入 JSON 文件的辅助类。"""

    @staticmethod
    def write(path: Path, payload: dict[str, Any]) -> None:
        """通过临时文件和原子替换写入 JSON。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
