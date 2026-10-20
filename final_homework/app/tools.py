"""Agent 工具定义、参数校验和统一执行注册表。"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .knowledge_base import PolicyRetriever
from .models import SourceReference


class SearchPolicyArgs(BaseModel):
    """``search_policy`` 工具的参数模型。"""

    query: str = Field(min_length=1, max_length=1000)


class CalculateLoanArgs(BaseModel):
    """``calculate_loan`` 工具的参数模型。"""

    principal: float = Field(gt=0)
    annual_rate: float = Field(ge=0, le=100)
    years: float = Field(gt=0, le=50)

    @field_validator("annual_rate")
    @classmethod
    def normalize_rate(cls, value: float) -> float:
        """同时兼容 ``3.5`` 和 ``0.035`` 两种年利率写法。"""
        if value > 1:
            return value / 100
        return value


@dataclass(slots=True)
class ToolResult:
    """所有工具统一返回的结果结构。"""

    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class BaseAgentTool(ABC):
    """Agent 工具的基础抽象接口。"""

    name: str
    description: str
    args_schema: type[BaseModel]

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """使用经过校验的关键字参数执行工具。"""

    def schema(self) -> dict[str, Any]:
        """返回供 LLM 工具规划使用的 JSON Schema。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }


class SearchPolicyTool(BaseAgentTool):
    """制度语义检索工具。"""

    name = "search_policy"
    description = "根据问题检索银行信贷制度、合规手册等内部规范，返回 Top3 来源条款。"
    args_schema = SearchPolicyArgs

    def __init__(self, retriever: PolicyRetriever) -> None:
        self.retriever = retriever

    def run(self, **kwargs: Any) -> ToolResult:
        """检索制度分块并返回可直接引用的来源。"""
        try:
            args = self.args_schema.model_validate(kwargs)
            sources = self.retriever.retrieve_sources(args.query, top_k=3)
            return ToolResult(
                name=self.name,
                ok=True,
                data={
                    "query": args.query,
                    "results": [source.model_dump() for source in sources],
                },
            )
        except Exception as exc:
            # 工具异常转换为标准结果，避免单个工具失败中断整个 LangGraph。
            return ToolResult(name=self.name, ok=False, error=str(exc))


class LoanCalculatorTool(BaseAgentTool):
    """等额本息贷款计算工具。"""

    name = "calculate_loan"
    description = "按等额本息计算贷款月供、总利息和还款总额。"
    args_schema = CalculateLoanArgs

    def run(self, **kwargs: Any) -> ToolResult:
        """计算月供、总利息和还款总额。"""
        try:
            args = self.args_schema.model_validate(kwargs)
            # 金融计算统一使用 Decimal，降低浮点数累计误差。
            principal = Decimal(str(args.principal))
            monthly_rate = Decimal(str(args.annual_rate)) / Decimal("12")
            months = int(
                (Decimal(str(args.years)) * Decimal("12")).quantize(
                    Decimal("1"),
                    rounding=ROUND_HALF_UP,
                )
            )
            if monthly_rate == 0:
                monthly_payment = principal / Decimal(months)
            else:
                # 等额本息公式：P × r × (1+r)^n / ((1+r)^n-1)。
                factor = (Decimal("1") + monthly_rate) ** months
                monthly_payment = (
                    principal
                    * monthly_rate
                    * factor
                    / (factor - Decimal("1"))
                )
            total_payment = monthly_payment * Decimal(months)
            total_interest = total_payment - principal
            money = Decimal("0.01")
            result = {
                "principal": float(principal),
                "annual_rate": float(args.annual_rate),
                "years": float(args.years),
                "months": months,
                "monthly_payment": float(
                    monthly_payment.quantize(money, rounding=ROUND_HALF_UP)
                ),
                "total_interest": float(
                    total_interest.quantize(money, rounding=ROUND_HALF_UP)
                ),
                "total_payment": float(
                    total_payment.quantize(money, rounding=ROUND_HALF_UP)
                ),
            }
            return ToolResult(name=self.name, ok=True, data=result)
        except Exception as exc:
            return ToolResult(name=self.name, ok=False, error=str(exc))


class ToolRegistry:
    """通过统一接口注册和执行工具。"""

    def __init__(self, tools: list[BaseAgentTool] | None = None) -> None:
        self._tools: dict[str, BaseAgentTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseAgentTool) -> None:
        """注册工具，并拒绝同名重复注册。"""
        if tool.name in self._tools:
            raise ValueError(f"工具已注册: {tool.name}")
        self._tools[tool.name] = tool

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """按名称执行工具并返回标准结果。"""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(name=name, ok=False, error=f"未知工具: {name}")
        return tool.run(**arguments)

    def schemas(self) -> list[dict[str, Any]]:
        """返回全部已注册工具的 Schema。"""
        return [tool.schema() for tool in self._tools.values()]

    def contains(self, name: str) -> bool:
        """判断指定工具是否已注册。"""
        return name in self._tools


@dataclass(slots=True)
class ToolCall:
    """一次计划中的工具调用。"""

    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolPlan:
    """一次请求对应的工具调用计划。"""

    calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的结构。"""
        return {"calls": [asdict(call) for call in self.calls]}


def heuristic_tool_plan(query: str) -> ToolPlan:
    """当模型不可用或规划失败时，根据中文关键词生成确定性工具计划。"""
    calls: list[ToolCall] = []
    calculation = extract_loan_arguments(query)
    if calculation:
        calls.append(ToolCall(name="calculate_loan", arguments=calculation))

    policy_keywords = (
        "制度",
        "规定",
        "政策",
        "流程",
        "标准",
        "认定",
        "审批",
        "合规",
        "要求",
        "期限",
        "材料",
        "条款",
    )
    if not calculation or any(keyword in query for keyword in policy_keywords):
        # 非计算问题默认检索制度；计算问题同时包含制度关键词时也执行检索。
        calls.append(
            ToolCall(
                name="search_policy",
                arguments={"query": query},
            )
        )
    return ToolPlan(calls=calls)


def extract_loan_arguments(query: str) -> dict[str, float] | None:
    """从自然语言问题中提取本金、年利率和贷款期限。"""
    principal_match = re.search(
        r"(?:贷款|本金)?\s*(\d+(?:\.\d+)?)\s*(亿元|亿|万元|万|元)\s*(?:贷款|本金)?",
        query,
    )
    rate_match = re.search(
        r"(?:年利率|年化利率|利率)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*%",
        query,
    )
    years_match = re.search(
        r"(?:期限|贷款期限|年限)\s*(?:为|是|[:：])?\s*(\d+(?:\.\d+)?)\s*年",
        query,
    )
    if not (principal_match and rate_match and years_match):
        return None

    amount = Decimal(principal_match.group(1))
    unit = principal_match.group(2)
    multipliers = {
        # 将中文金额单位统一换算为“元”。
        "亿元": Decimal("100000000"),
        "亿": Decimal("100000000"),
        "万元": Decimal("10000"),
        "万": Decimal("10000"),
        "元": Decimal("1"),
    }
    principal = amount * multipliers[unit]
    return {
        "principal": float(principal),
        "annual_rate": float(rate_match.group(1)),
        "years": float(years_match.group(1)),
    }
