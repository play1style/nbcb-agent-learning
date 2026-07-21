"""数据模型模块——定义 API 请求/响应的 Pydantic 数据结构。"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """用户发起的合规咨询请求。"""
    query: str = Field(..., description="用户查询内容")
    user_id: str = Field(default="anonymous", description="用户标识")


class Source(BaseModel):
    """答案中某一个事实的来源归属。"""
    document: str = Field(..., description="来源文档名称")
    chapter: str = Field(..., description="章节号")
    content: str = Field(..., description="相关内容片段")


class ToolCall(BaseModel):
    """一次工具调用的记录。"""
    tool: str = Field(..., description="工具名称")
    args: Dict[str, Any] = Field(default_factory=dict, description="工具参数")


class ChatResponse(BaseModel):
    """返回给客户端的完整响应。"""
    answer: str = Field(..., description="助手回答")
    sources: List[Source] = Field(default_factory=list, description="来源引用")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="工具调用链")
    processing_time_ms: int = Field(..., description="处理耗时（毫秒）")


class AuditLogEntry(BaseModel):
    """持久化的审计日志条目。"""
    user_id: str
    query: str
    sanitized_query: str
    tool_calls: List[Dict[str, Any]]
    response: str
    processing_time_ms: int
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
