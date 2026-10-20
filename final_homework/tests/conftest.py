"""共享 pytest 测试夹具：全部使用离线组件，不调用外部模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.agent import ComplianceAgent
from app.config import PROJECT_ROOT, Settings
from app.embeddings import HashEmbeddingClient
from app.knowledge_base import PolicyRetriever, VectorStore
from app.main import create_app
from app.tools import ToolPlan, heuristic_tool_plan


class SilentLLM:
    """禁用外部生成的确定性 LLM 测试替身。"""

    available = False

    def plan_tools(
        self,
        query: str,
        history: str,
        tool_schemas: list[dict[str, Any]],
    ) -> ToolPlan:
        """使用确定性规则选择工具。"""
        return heuristic_tool_plan(query)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """返回空字符串，以覆盖带引用的安全兜底答案逻辑。"""
        return ""


@pytest.fixture()
def runtime(tmp_path: Path) -> dict[str, Any]:
    """构建相互隔离的 Agent、向量库和 API 测试环境。"""
    settings = Settings(
        knowledge_dir=PROJECT_ROOT / "knowledge",
        vector_store_path=tmp_path / "vector_store.json",
        data_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        deepseek_api_key="",
        embedding_provider="hash",
        retrieval_threshold=0.48,
        medium_confidence_threshold=0.48,
    )
    settings.ensure_directories()
    embedder = HashEmbeddingClient(settings.embedding_dimension)
    vector_store = VectorStore(settings, embedder)
    vector_store.ensure_index(force=True)
    agent = ComplianceAgent(
        settings=settings,
        retriever=PolicyRetriever(vector_store),
        llm=SilentLLM(),
    )
    app = create_app(settings=settings, agent=agent)
    return {
        "settings": settings,
        "embedder": embedder,
        "vector_store": vector_store,
        "agent": agent,
        "app": app,
    }


@pytest.fixture()
def client(runtime: dict[str, Any]):
    """创建启用 FastAPI lifespan 的 TestClient。"""
    with TestClient(runtime["app"]) as test_client:
        yield test_client
