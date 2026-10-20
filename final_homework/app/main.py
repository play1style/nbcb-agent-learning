"""企业级智能合规助手的 FastAPI 应用入口。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from .agent import ComplianceAgent
from .config import Settings
from .embeddings import create_embedder
from .knowledge_base import PolicyRetriever, VectorStore
from .models import (
    ChatRequest,
    ChatResponse,
    MemoryResponse,
    PolicySearchResponse,
    ReindexResponse,
)


logging.basicConfig(
    # 统一日志格式，使 LangGraph 节点和远端模型调用日志可直接观察。
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def create_app(
    settings: Settings | None = None,
    agent: ComplianceAgent | None = None,
) -> FastAPI:
    """创建 FastAPI 应用，并支持测试时注入依赖。"""
    runtime_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        """服务启动前初始化向量索引和 Agent。"""
        runtime_settings.ensure_directories()
        if application.state.agent is None:
            # 启动阶段可能涉及网络和磁盘操作，放入线程池避免阻塞事件循环。
            embedder = create_embedder(runtime_settings)
            vector_store = VectorStore(runtime_settings, embedder)
            await run_in_threadpool(vector_store.ensure_index)
            application.state.vector_store = vector_store
            application.state.agent = ComplianceAgent(
                settings=runtime_settings,
                retriever=PolicyRetriever(vector_store),
            )
        yield

    application = FastAPI(
        title=runtime_settings.app_name,
        version="1.0.0",
        description=(
            "基于 LangGraph、RAG、可溯源引用、持久化记忆和审计链的"
            "银行智能合规助手。"
        ),
        lifespan=lifespan,
    )
    application.state.settings = runtime_settings
    application.state.agent = agent
    application.state.vector_store = (
        agent.retriever.vector_store if agent else None
    )

    @application.get("/", tags=["system"])
    async def root() -> dict[str, str]:
        """返回服务基础元数据。"""
        return {
            "service": runtime_settings.app_name,
            "version": "1.0.0",
            "docs": "/docs",
        }

    @application.get("/health", tags=["system"])
    async def health(request: Request) -> dict[str, Any]:
        """返回服务、向量索引和模型配置健康状态。"""
        vector_store: VectorStore | None = request.app.state.vector_store
        runtime_agent: ComplianceAgent | None = request.app.state.agent
        llm_settings = (
            getattr(runtime_agent.llm, "settings", None)
            if runtime_agent
            else None
        )
        return {
            "status": "ok" if runtime_agent else "unavailable",
            "environment": runtime_settings.environment,
            "embedding_backend": (
                vector_store.embedder.identity if vector_store else "unknown"
            ),
            "indexed_documents": (
                vector_store.metadata.get("document_count", 0)
                if vector_store
                else 0
            ),
            "indexed_chunks": (
                vector_store.metadata.get("chunk_count", 0)
                if vector_store
                else 0
            ),
            "llm_available": bool(
                runtime_agent and runtime_agent.llm.available
            ),
            "llm_model": (
                getattr(llm_settings, "deepseek_model", "")
                if llm_settings
                else ""
            ),
            "llm_last_error": (
                getattr(runtime_agent.llm, "last_error", "")
                if runtime_agent
                else ""
            ),
        }

    @application.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        tags=["chat"],
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        """执行一轮可持久化、可审计的合规对话。"""
        runtime_agent: ComplianceAgent | None = request.app.state.agent
        if runtime_agent is None:
            raise HTTPException(status_code=503, detail="服务尚未初始化")
        # Agent 内部包含同步文件、Embedding 和 LLM 调用，放到线程池执行。
        return await run_in_threadpool(
            runtime_agent.invoke,
            payload.message,
            payload.session_id,
            payload.user_id,
        )

    @application.get(
        "/api/v1/policies/search",
        response_model=PolicySearchResponse,
        tags=["knowledge"],
    )
    async def search_policies(
        request: Request,
        query: str = Query(min_length=1, max_length=1000),
        top_k: int = Query(default=3, ge=1, le=20),
    ) -> PolicySearchResponse:
        """通过 Embedding 语义相似度检索制度章节。"""
        vector_store: VectorStore | None = request.app.state.vector_store
        if vector_store is None:
            raise HTTPException(status_code=503, detail="索引尚未初始化")
        results = await run_in_threadpool(
            vector_store.source_references,
            query,
            top_k,
        )
        return PolicySearchResponse(query=query, results=results)

    @application.post(
        "/api/v1/knowledge/reindex",
        response_model=ReindexResponse,
        tags=["knowledge"],
    )
    async def reindex(request: Request) -> ReindexResponse:
        """强制重新加载 Markdown 并重建向量索引。"""
        vector_store: VectorStore | None = request.app.state.vector_store
        if vector_store is None:
            raise HTTPException(status_code=503, detail="索引尚未初始化")
        await run_in_threadpool(vector_store.build, True)
        return ReindexResponse(
            document_count=vector_store.metadata["document_count"],
            chunk_count=vector_store.metadata["chunk_count"],
            embedding_backend=vector_store.embedder.identity,
            vector_store_path=str(vector_store.path),
            rebuilt=True,
        )

    @application.get(
        "/api/v1/sessions/{session_id}/memory",
        response_model=MemoryResponse,
        tags=["memory"],
    )
    async def get_memory(
        session_id: str,
        request: Request,
        user_id: str = Query(default="anonymous", min_length=1, max_length=128),
        limit_turns: int = Query(default=20, ge=1, le=100),
    ) -> MemoryResponse:
        """读取指定用户和会话的持久化消息。"""
        runtime_agent: ComplianceAgent | None = request.app.state.agent
        if runtime_agent is None:
            raise HTTPException(status_code=503, detail="服务尚未初始化")
        try:
            messages = await run_in_threadpool(
                runtime_agent.memory.load_messages,
                session_id,
                user_id,
                limit_turns,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return MemoryResponse(session_id=session_id, messages=messages)

    @application.get("/api/v1/admin/audit", tags=["audit"])
    async def read_audit(
        request: Request,
        user_id: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        """读取近期审计记录，供合规复核。"""
        runtime_agent: ComplianceAgent | None = request.app.state.agent
        if runtime_agent is None:
            raise HTTPException(status_code=503, detail="服务尚未初始化")
        records = await run_in_threadpool(
            runtime_agent.audit_logger.read_recent,
            user_id,
            limit,
        )
        return {"count": len(records), "records": records}

    return application


app = create_app()


def run() -> None:
    """使用 Uvicorn 启动 API 服务。"""
    settings = Settings.from_env()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
