"""API 请求、响应和领域数据模型。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ConfidenceLevel = Literal["高", "中", "低"]


class ChatRequest(BaseModel):
    """对话接口请求。"""

    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default", min_length=1, max_length=64)
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        """限制会话 ID 字符，防止路径穿越。"""
        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789_-"
        )
        if not value or any(char not in allowed for char in value):
            raise ValueError(
                "session_id 仅允许字母、数字、下划线和连字符，且不能为空"
            )
        return value


class SourceReference(BaseModel):
    """答案引用的制度来源。"""

    document: str
    section: str
    content: str
    score: float = Field(ge=0.0, le=1.0)

    @property
    def citation(self) -> str:
        """返回产品要求的 ``[来源文档:章节号]`` 引用格式。"""
        return f"[{self.document}:{self.section}]"


class ChatResponse(BaseModel):
    """对话接口响应。"""

    request_id: str
    session_id: str
    answer: str
    confidence: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    sources: list[SourceReference] = Field(default_factory=list)
    tool_chain: list[str] = Field(default_factory=list)
    blocked: bool = False
    latency_ms: int = Field(ge=0)


class PolicySearchResponse(BaseModel):
    """制度检索接口响应。"""

    query: str
    results: list[SourceReference]


class ReindexResponse(BaseModel):
    """向量索引重建结果。"""

    document_count: int
    chunk_count: int
    embedding_backend: str
    vector_store_path: str
    rebuilt: bool


class MemoryMessage(BaseModel):
    """持久化的单条会话消息。"""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    role: Literal["user", "assistant"]
    content: str
    user_id: str
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """会话记忆查询响应。"""

    session_id: str
    messages: list[MemoryMessage]
