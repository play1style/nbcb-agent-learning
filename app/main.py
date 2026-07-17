"""FastAPI主应用入口.

提供合规智能助手的REST API接口，包含聊天、健康检查等端点。
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent.agent import agent
from app.config import config
from app.models.schemas import (
    AuditLogEntry,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from app.security.audit_logger import audit_logger

# 创建FastAPI应用
app = FastAPI(
    title="银行合规智能助手 API",
    description="基于AI的银行合规岗位辅助系统，"
    "支持制度查询、贷款计算等功能",
    version="1.0.0",
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """全局异常处理器.

    Args:
        request: 请求对象
        exc: 异常对象

    Returns:
        错误响应
    """
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=f"服务器内部错误: {str(exc)}",
            code="INTERNAL_ERROR",
        ).model_dump(),
    )


@app.get("/health", tags=["系统"])
async def health_check() -> dict:
    """健康检查接口.

    Returns:
        服务状态信息
    """
    return {
        "status": "ok",
        "service": "银行合规智能助手",
        "version": "1.0.0",
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["对话"])
async def chat(request: ChatRequest) -> ChatResponse:
    """智能体对话接口.

    接收用户查询，经安全检测、意图识别、工具调用后返回结构化回答。

    Args:
        request: 包含query和user_id的请求体

    Returns:
        包含回答、来源、工具调用链等信息的响应

    Raises:
        HTTPException: 处理失败
    """
    try:
        # 处理查询
        response = agent.process(
            query=request.query,
            user_id=request.user_id,
        )

        # 记录审计日志
        tool_chain_log = [
            {
                "tool": tc.tool_name,
                "args": tc.arguments,
            }
            for tc in response.tool_calls
        ]

        audit_logger.log(
            user_id=request.user_id,
            query=request.query,
            tool_chain=tool_chain_log,
            response_time_ms=response.processing_time_ms,
            injection_detected=response.injection_detected,
            answer_preview=response.answer[:200],
        )


        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时出错: {str(e)}",
        )


@app.get("/api/audit/recent", tags=["审计"])
async def get_recent_audit_logs(count: int = 10) -> list[AuditLogEntry]:
    """获取最近审计日志.

    Args:
        count: 返回条目数

    Returns:
        最近的审计日志列表
    """
    logs = audit_logger.get_recent_logs(count=count)
    return [AuditLogEntry(**log) for log in logs]


@app.get("/api/tools", tags=["工具"])
async def list_tools() -> dict:
    """列出所有可用工具.

    Returns:
        工具列表
    """
    from app.tools.registry import tool_registry

    tools_info = {}
    for name in tool_registry.list_tools():
        tool = tool_registry.get_tool(name)
        if tool:
            tools_info[name] = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
    return {"tools": tools_info}


def start() -> None:
    """启动服务（用于直接运行）. """
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
    )


if __name__ == "__main__":
    start()
