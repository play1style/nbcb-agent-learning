"""Pydantic数据模型定义.

定义API请求/响应结构、知识库条目模型等。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeRule(BaseModel):
    """知识库规则条目模型.

    Attributes:
        id: 规则唯一标识
        document: 来源文档名称
        section: 章节号
        title: 规则标题
        content: 规则内容片段
    """

    id: int = Field(description="规则唯一标识")
    document: str = Field(description="来源文档名称")
    section: str = Field(description="章节号")
    title: str = Field(description="规则标题")
    content: str = Field(description="规则内容片段")


class SearchResult(BaseModel):
    """检索结果条目模型.

    Attributes:
        rule: 匹配的规则条目
        similarity: 相似度得分
        source_tag: 溯源标签，格式 [文档名:章节号]
    """

    rule: KnowledgeRule
    similarity: float = Field(description="相似度得分", ge=0.0, le=1.0)
    source_tag: str = Field(description="溯源标签，格式 [文档名:章节号]")


class ChatRequest(BaseModel):
    """聊天请求模型.

    Attributes:
        query: 用户查询文本
        user_id: 用户标识，用于审计日志
    """

    query: str = Field(
        ..., min_length=1, max_length=2000, description="用户查询文本"
    )
    user_id: str = Field(
        default="anonymous", description="用户标识，用于审计日志"
    )


class ToolCallInfo(BaseModel):
    """工具调用信息模型.

    Attributes:
        tool_name: 工具名称
        arguments: 调用参数
        result: 调用结果摘要
    """

    tool_name: str = Field(description="工具名称")
    arguments: dict[str, Any] = Field(description="调用参数")
    result: str = Field(description="调用结果摘要")


class ChatResponse(BaseModel):
    """聊天响应模型.

    Attributes:
        answer: 最终回答文本
        sources: 引用来源列表
        tool_calls: 工具调用链
        processing_time_ms: 处理耗时（毫秒）
        injection_detected: 是否检测到注入攻击
    """

    answer: str = Field(description="最终回答文本")
    sources: list[str] = Field(
        default_factory=list, description="引用来源列表"
    )
    tool_calls: list[ToolCallInfo] = Field(
        default_factory=list, description="工具调用链"
    )
    processing_time_ms: float = Field(
        description="处理耗时（毫秒）"
    )
    injection_detected: bool = Field(
        default=False, description="是否检测到注入攻击"
    )


class AuditLogEntry(BaseModel):
    """审计日志条目模型.

    Attributes:
        timestamp: 时间戳
        user_id: 用户标识
        query: 查询内容
        tool_chain: 工具调用链摘要
        response_time_ms: 响应耗时
        injection_detected: 是否检测到注入
        answer_preview: 回答摘要
    """

    timestamp: datetime = Field(description="时间戳")
    user_id: str = Field(description="用户标识")
    query: str = Field(description="查询内容")
    tool_chain: list[dict[str, Any]] = Field(
        default_factory=list, description="工具调用链摘要"
    )
    response_time_ms: float = Field(description="响应耗时")
    injection_detected: bool = Field(
        default=False, description="是否检测到注入"
    )
    answer_preview: str = Field(
        default="", description="回答摘要"
    )


class ErrorResponse(BaseModel):
    """错误响应模型.

    Attributes:
        detail: 错误描述
        code: 错误码
    """

    detail: str = Field(description="错误描述")
    code: str = Field(default="INTERNAL_ERROR", description="错误码")
