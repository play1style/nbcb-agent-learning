"""企业级智能合规助手的 LangGraph 编排核心。"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .audit import AuditLogger, AuditRecord
from .config import Settings
from .knowledge_base import PolicyRetriever, VectorStore
from .llm import LLMService, SYSTEM_PROMPT, build_answer_prompt
from .memory import ComplianceBackupStore, ConversationMemory
from .models import ChatResponse, MemoryMessage, SourceReference
from .security import injection_response, redact_text, sanitize_input
from .tools import (
    BaseAgentTool,
    LoanCalculatorTool,
    SearchPolicyTool,
    ToolPlan,
    ToolRegistry,
)


logger = logging.getLogger(__name__)


class ComplianceAgentState(TypedDict, total=False):
    """在 LangGraph 节点之间传递的共享状态。"""

    request_id: str
    user_id: str
    session_id: str
    query: str
    sanitized_query: str
    blocked: bool
    block_reason: str
    started_perf: float
    memory_messages: list[MemoryMessage]
    memory_context: str
    tool_plan: ToolPlan
    tool_results: list[dict[str, Any]]
    tool_chain: list[str]
    sources: list[SourceReference]
    calculation: dict[str, Any] | None
    answer: str
    confidence: str
    confidence_score: float
    response: ChatResponse


def _citation_set(text: str) -> set[str]:
    """从答案文本中提取 ``[来源文档:章节号]`` 引用。"""
    return set(re.findall(r"\[[^\[\]:：]{1,80}[:：][^\[\]]{1,50}\]", text))


def _format_number(value: float) -> str:
    """将金额格式化为带千位分隔符的两位小数。"""
    return f"{value:,.2f}"


def format_loan_calculation(data: dict[str, Any]) -> str:
    """将贷款计算工具结果格式化为可审计的中文答案。"""
    return (
        "等额本息计算结果："
        f"贷款本金 {_format_number(data['principal'])} 元，"
        f"月供 {_format_number(data['monthly_payment'])} 元，"
        f"总利息 {_format_number(data['total_interest'])} 元，"
        f"还款总额 {_format_number(data['total_payment'])} 元。"
        f"（年利率 {data['annual_rate'] * 100:.2f}%，期限 {data['years']:g} 年）"
    )


class ComplianceAgent:
    """基于 LangGraph 和 RAG 的智能合规助手。"""

    def __init__(
        self,
        settings: Settings,
        retriever: PolicyRetriever,
        llm: LLMService | None = None,
        memory: ConversationMemory | None = None,
        backup_store: ComplianceBackupStore | None = None,
        audit_logger: AuditLogger | None = None,
        tools: list[BaseAgentTool] | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm or LLMService(settings)
        self.memory = memory or ConversationMemory(
            settings.data_dir,
            settings.memory_window_turns,
        )
        self.backup_store = backup_store or ComplianceBackupStore(
            settings.data_dir
        )
        self.audit_logger = audit_logger or AuditLogger(settings.log_dir)
        registered_tools = tools or [
            SearchPolicyTool(retriever),
            LoanCalculatorTool(),
        ]
        self.tool_registry = ToolRegistry(registered_tools)
        self.graph = self._build_graph()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        embedder=None,
        llm: LLMService | None = None,
    ) -> "ComplianceAgent":
        """构建生产 Agent，并确保向量索引已经可用。"""
        from .embeddings import create_embedder

        settings.ensure_directories()
        embedder = embedder or create_embedder(settings)
        vector_store = VectorStore(settings, embedder)
        vector_store.ensure_index()
        return cls(
            settings=settings,
            retriever=PolicyRetriever(vector_store),
            llm=llm,
        )

    def invoke(
        self,
        message: str,
        session_id: str = "default",
        user_id: str = "anonymous",
    ) -> ChatResponse:
        """通过 LangGraph 执行一轮完整对话。"""
        initial_state: ComplianceAgentState = {
            "request_id": str(uuid.uuid4()),
            "user_id": user_id,
            "session_id": session_id,
            "query": message,
            "started_perf": time.perf_counter(),
            "tool_results": [],
            "tool_chain": [],
            "sources": [],
            "calculation": None,
        }
        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 25},
        )
        return final_state["response"]

    def _build_graph(self):
        """创建并编译显式合规工作流。"""
        graph = StateGraph(ComplianceAgentState)
        # 每个节点只负责一个明确阶段，便于测试、审计和替换实现。
        graph.add_node("input_guard", self._input_guard_node)
        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("plan_tools", self._plan_tools_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.add_node("generate_answer", self._generate_answer_node)
        graph.add_node("blocked_answer", self._blocked_answer_node)
        graph.add_node("finalize", self._finalize_node)
        graph.add_node("persist_turn", self._persist_turn_node)
        graph.add_node("audit", self._audit_node)

        graph.add_edge(START, "input_guard")
        # 命中注入攻击时跳过记忆、工具和模型调用，直接生成安全回复。
        graph.add_conditional_edges(
            "input_guard",
            self._route_after_guard,
            {
                "blocked": "blocked_answer",
                "continue": "load_memory",
            },
        )
        graph.add_edge("load_memory", "plan_tools")
        graph.add_edge("plan_tools", "execute_tools")
        graph.add_edge("execute_tools", "generate_answer")
        graph.add_edge("generate_answer", "finalize")
        graph.add_edge("blocked_answer", "finalize")
        # 所有正常和拦截路径最终都会经过持久化和审计节点。
        graph.add_edge("finalize", "persist_turn")
        graph.add_edge("persist_turn", "audit")
        graph.add_edge("audit", END)
        return graph.compile()

    @staticmethod
    def _log_node(
        node_name: str,
        state: ComplianceAgentState,
        **details: Any,
    ) -> None:
        """记录 LangGraph 节点被调用，但不输出原始用户问题。"""
        context = " ".join(
            f"{key}={value}" for key, value in details.items()
        )
        logger.info(
            "LangGraph node invoked: node=%s request_id=%s "
            "session_id=%s user_id=%s%s",
            node_name,
            state.get("request_id", ""),
            state.get("session_id", ""),
            state.get("user_id", ""),
            f" {context}" if context else "",
        )

    def _input_guard_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """净化输入并拦截提示词注入。"""
        self._log_node("input_guard", state)
        result = sanitize_input(
            state.get("query", ""),
            max_length=self.settings.max_input_length,
        )
        blocked = result.blocked or not result.text
        reason = result.reason or (not result.text and "输入不能为空") or ""
        return {
            "sanitized_query": result.text,
            "blocked": blocked,
            "block_reason": reason,
        }

    @staticmethod
    def _route_after_guard(state: ComplianceAgentState) -> str:
        """将已拦截请求路由到安全回答节点。"""
        return "blocked" if state.get("blocked") else "continue"

    def _load_memory_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """读取会话记忆并自动渲染为后续提示词上下文。"""
        self._log_node("load_memory", state)
        messages = self.memory.load_messages(
            state["session_id"],
            user_id=state["user_id"],
            limit_turns=self.settings.memory_window_turns,
        )
        return {
            "memory_messages": messages,
            "memory_context": self.memory.render_context(messages),
        }

    def _plan_tools_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """优先让模型规划工具，并使用确定性规则兜底。"""
        self._log_node("plan_tools", state)
        plan = self.llm.plan_tools(
            query=state["sanitized_query"],
            history=state.get("memory_context", ""),
            tool_schemas=self.tool_registry.schemas(),
        )
        return {"tool_plan": plan}

    def _execute_tools_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """通过统一注册表执行全部计划工具。"""
        plan = state.get("tool_plan", ToolPlan())
        self._log_node(
            "execute_tools",
            state,
            planned_tools=len(plan.calls),
        )
        results: list[dict[str, Any]] = []
        trace: list[str] = []
        sources: list[SourceReference] = []
        calculation: dict[str, Any] | None = None
        for call in plan.calls:
            result = self.tool_registry.execute(call.name, call.arguments)
            record = {
                "name": result.name,
                "ok": result.ok,
                "data": result.data,
                "error": result.error,
            }
            results.append(record)
            arguments = ", ".join(
                f"{key}={value}" for key, value in call.arguments.items()
            )
            trace.append(
                f"{call.name}({arguments})"
                + ("" if result.ok else f"[error={result.error}]")
            )
            if result.ok and result.name == "search_policy":
                # 检索工具返回的制度来源将进入答案生成和溯源阶段。
                sources.extend(
                    SourceReference.model_validate(item)
                    for item in result.data.get("results", [])
                )
            if result.ok and result.name == "calculate_loan":
                # 贷款结果保留为结构化数据，避免再次解析自然语言。
                calculation = result.data
        return {
            "tool_results": results,
            "tool_chain": trace,
            "sources": sources,
            "calculation": calculation,
        }

    def _generate_answer_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """生成受知识库约束的答案，并计算置信度。"""
        self._log_node(
            "generate_answer",
            state,
            retrieved_sources=len(state.get("sources", [])),
        )
        sources = sorted(
            state.get("sources", []),
            key=lambda source: source.score,
            reverse=True,
        )[: self.settings.retrieval_top_k]
        active_sources = [
            source
            for source in sources
            if source.score >= self.settings.retrieval_threshold
        ]
        calculation = state.get("calculation")

        if not active_sources and not calculation:
            # 没有达到阈值的制度来源，也没有计算结果时不调用 LLM，防止臆答。
            logger.info(
                "LLM generation skipped: request_id=%s reason=no_active_sources "
                "top_score=%s threshold=%s",
                state["request_id"],
                sources[0].score if sources else 0.0,
                self.settings.retrieval_threshold,
            )
            return {
                "answer": (
                    "根据现有知识库无法确认，请咨询相关部门"
                ),
                "confidence": "低",
                "confidence_score": (
                    sources[0].score if sources else 0.0
                ),
                "sources": [],
            }

        if calculation and not active_sources:
            # 纯计算问题使用确定性模板，保证金额结果稳定可复现。
            return {
                "answer": format_loan_calculation(calculation),
                "confidence": "高",
                "confidence_score": 1.0,
                "sources": [],
            }

        prompt = build_answer_prompt(
            query=state["sanitized_query"],
            memory_context=state.get("memory_context", ""),
            sources=active_sources,
            calculation=calculation,
        )
        generated = self.llm.chat(SYSTEM_PROMPT, prompt)
        valid_citations = {
            source.citation for source in active_sources
        }
        if (
            generated
            and _citation_set(generated) & valid_citations
        ):
            # 模型答案至少包含一个有效来源引用才允许直接返回。
            answer = generated
        else:
            # 模型不可用或缺少引用时，回退为由制度原文和引用拼装的答案。
            answer = self._fallback_grounded_answer(
                active_sources,
                calculation,
            )

        top_score = active_sources[0].score
        confidence, confidence_score = self._confidence_from_score(top_score)
        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "sources": active_sources,
        }

    def _blocked_answer_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """返回提示词注入被拦截时的安全回复。"""
        self._log_node(
            "blocked_answer",
            state,
            reason=state.get("block_reason", ""),
        )
        return {
            "answer": injection_response(),
            "confidence": "高",
            "confidence_score": 1.0,
            "sources": [],
        }

    def _finalize_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """脱敏输出、追加置信度并构建 API 响应。"""
        self._log_node(
            "finalize",
            state,
            confidence=state.get("confidence", "低"),
        )
        answer = redact_text(state.get("answer", "").strip())
        confidence = state.get("confidence", "低")
        # 所有回答统一追加置信度，供合规人员判断是否需要人工复核。
        answer = f"{answer}\n\n置信度：{confidence}"
        if confidence == "低" and (
            "根据现有知识库无法确认，请咨询相关部门" not in answer
        ):
            answer += "\n根据现有知识库无法确认，请咨询相关部门"
        latency_ms = max(
            0,
            int((time.perf_counter() - state["started_perf"]) * 1000),
        )
        response = ChatResponse(
            request_id=state["request_id"],
            session_id=state["session_id"],
            answer=answer,
            confidence=confidence,
            confidence_score=state.get("confidence_score", 0.0),
            sources=state.get("sources", []),
            tool_chain=state.get("tool_chain", []),
            blocked=state.get("blocked", False),
            latency_ms=latency_ms,
        )
        return {"response": response}

    def _persist_turn_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """持久化会话记忆和哈希链合规备份。"""
        self._log_node("persist_turn", state)
        response = state["response"]
        now = datetime.now(timezone.utc)
        user_message = MemoryMessage(
            timestamp=now,
            role="user",
            content=redact_text(state.get("sanitized_query", "")),
            user_id=state["user_id"],
            request_id=state["request_id"],
            metadata={"blocked": state.get("blocked", False)},
        )
        assistant_message = MemoryMessage(
            timestamp=now,
            role="assistant",
            content=response.answer,
            user_id=state["user_id"],
            request_id=state["request_id"],
            metadata={
                "confidence": response.confidence,
                "sources": [source.citation for source in response.sources],
            },
        )
        self.memory.append_turn(
            state["session_id"],
            user_message,
            assistant_message,
        )
        self.backup_store.append(
            # 备份保留工具链、来源和置信度，便于后续合规追溯。
            session_id=state["session_id"],
            record={
                "request_id": state["request_id"],
                "user_id": state["user_id"],
                "query": user_message.content,
                "answer": assistant_message.content,
                "tool_chain": response.tool_chain,
                "sources": [source.citation for source in response.sources],
                "confidence": response.confidence,
                "confidence_score": response.confidence_score,
                "blocked": response.blocked,
                "latency_ms": response.latency_ms,
            },
        )
        return {}

    def _audit_node(
        self,
        state: ComplianceAgentState,
    ) -> ComplianceAgentState:
        """记录用户、查询、工具链、来源、耗时和响应哈希。"""
        self._log_node(
            "audit",
            state,
            tool_calls=len(state["response"].tool_chain),
        )
        response = state["response"]
        query = redact_text(state.get("sanitized_query", ""))
        self.audit_logger.record(
            AuditRecord(
                request_id=state["request_id"],
                user_id=state["user_id"],
                session_id=state["session_id"],
                query=query,
                # 同时保存不可逆查询哈希，避免仅依赖可能被修改的文本。
                query_hash=hashlib.sha256(
                    state.get("sanitized_query", "").encode("utf-8")
                ).hexdigest(),
                tool_chain=response.tool_chain,
                sources=[source.citation for source in response.sources],
                confidence=response.confidence,
                confidence_score=response.confidence_score,
                blocked=response.blocked,
                latency_ms=response.latency_ms,
            ),
            response=response.answer,
        )
        return {}

    def _fallback_grounded_answer(
        self,
        sources: list[SourceReference],
        calculation: dict[str, Any] | None,
    ) -> str:
        """不依赖 LLM，基于制度原文构建带引用的安全答案。"""
        blocks: list[str] = []
        if calculation:
            blocks.append(format_loan_calculation(calculation))
        for source in sources:
            content = source.content.strip()
            blocks.append(f"{content} {source.citation}")
        return "\n\n".join(blocks)

    def _confidence_from_score(
        self,
        score: float,
    ) -> tuple[str, float]:
        """将检索分数映射为高、中、低三档置信度。"""
        if score >= self.settings.high_confidence_threshold:
            return "高", score
        if score >= self.settings.medium_confidence_threshold:
            return "中", score
        return "低", score
