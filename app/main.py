"""FastAPI 应用入口——启动 Web 服务，注册 API 路由。"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agent import ComplianceAgent
from .config import Config
from .knowledge_base import KnowledgeBase
from .models import ChatRequest, ChatResponse, Source, ToolCall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── 全局单例 ────────────────────────────────────────────────────────
_agent: ComplianceAgent | None = None


def get_agent() -> ComplianceAgent:
    """获取全局 Agent 实例（启动时初始化）。"""
    if _agent is None:
        raise RuntimeError("Agent 尚未初始化？")
    return _agent


# ── 应用生命周期 ───────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的生命周期管理——初始化知识库和 Agent。"""
    global _agent
    config = Config()

    logger.info("Loading knowledge base …")
    kb = KnowledgeBase(config.POLICY_FILE, config.EMBEDDING_MODEL, config=config)

    logger.info("Initialising compliance agent …")
    _agent = ComplianceAgent(kb, config)

    logger.info("Compliance assistant ready on %s:%s", config.HOST, config.PORT)
    yield

    _agent = None


# ── FastAPI 应用 ────────────────────────────────────────────────────

app = FastAPI(
    title="智能合规助手 API",
    description="基于 LangGraph 的企业级银行合规智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API 路由 ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """健康检查——返回 Agent 就绪状态。"""
    return {"status": "ok", "agent_ready": _agent is not None}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    提交合规相关问题。

    Agent 会自动决定是否检索制度文档、执行计算或直接作答。
    """
    agent = get_agent()
    try:
        result = agent.query(request.query, request.user_id)
        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            tool_calls=[ToolCall(**t) for t in result["tool_calls"]],
            processing_time_ms=result["processing_time_ms"],
        )
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail="内部处理错误，请稍后重试") from exc


@app.get("/api/audit/logs")
async def list_audit_logs(limit: int = 50):
    """返回最近 *limit* 条审计日志记录。"""
    agent = get_agent()
    logs = agent.audit_logger.get_logs(limit)
    return {"logs": logs, "count": len(logs)}
