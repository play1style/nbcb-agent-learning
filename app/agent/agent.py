"""智能体Agent模块.

实现ReAct（Reasoning + Acting）模式的智能体，负责意图识别、工具调用
和答案生成。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from app.agent.llm_client import llm_client
from app.models.schemas import (
    ChatResponse,
    ToolCallInfo,
)
from app.knowledge_base.data import knowledge_base
from app.security.input_sanitizer import input_sanitizer
from app.security.output_desensitizer import output_desensitizer
from app.tools.registry import tool_registry

# 注册工具（触发装饰器）
import app.tools.search_policy  # noqa: F401
import app.tools.calculate_loan  # noqa: F401

# ========== 知识库有效来源列表（防幻觉） ==========
_VALID_SOURCES: set[str] = set()
for _rule in knowledge_base.rules:
    _VALID_SOURCES.add(f"[{_rule.document}:{_rule.section}]")


# 系统提示词模板
SYSTEM_PROMPT = """你是银行合规智能助手，一个专业、严谨的银行合规岗位辅助AI系统。

## 核心职责
- 回答关于银行信贷制度、合规操作手册等内部规范的问题
- 协助计算贷款相关数据
- 所有回答必须基于已掌握的制度和数据，不得编造信息

## 回答规范
1. **溯源要求**：每个关键事实必须标注来源，格式为 [来源文档:章节号]，例如 [信贷业务制度:4.1]
2. **严禁编造来源**：只能引用 search_policy 工具实际返回的 source_tag 值，禁止编造任何不存在的文档名、章节号或来源格式
3. **信息不足时**：明确回复"根据现有知识库无法确认，请咨询相关部门"
4. **语言风格**：专业、简洁、清晰

## 当前知识库包含以下来源
{available_sources}

## 可用工具
{tool_descriptions}

## 行为准则
- 如果用户询问制度规定类问题，使用search_policy工具检索
- 如果用户询问贷款计算类问题，使用calculate_loan工具计算
- 如果同时涉及制度查询和计算，先检索再计算
- **引用来源的强制规则**：你只能在 search_policy 返回结果的 source_tag 字段中选择引用，严禁编造任何不存在的来源名称
- 如果检索结果与问题不相关或为空，直接回复"根据现有知识库无法确认，请咨询相关部门"
"""


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """从LLM响应文本中解析工具调用JSON.

    支持多种JSON格式，包括独立JSON块和混在文本中的JSON。

    Args:
        text: LLM响应文本

    Returns:
        解析出的工具调用，None表示无工具调用
    """
    # 尝试提取```json ... ```格式
    json_block = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
    )
    if json_block:
        try:
            parsed = json.loads(json_block.group(1))
            if "tool" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    # 尝试提取纯JSON对象
    json_obj = re.search(r"\{[^{}]*\"tool\"[^{}]*\}", text, re.DOTALL)
    if json_obj:
        try:
            parsed = json.loads(json_obj.group(0))
            if "tool" in parsed:
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

    return None


class ComplianceAgent:
    """银行合规智能体.

    基于ReAct模式实现，能够理解用户意图，调用工具获取信息，
    并基于工具结果生成带有来源标注的回答。

    Examples:
        >>> agent = ComplianceAgent()
        >>> response = agent.process("贷款审批流程是什么", user_id="test")
        >>> response.sources
        ['[信贷业务制度:4.1]']
    """

    def __init__(self) -> None:
        """初始化合规智能体。"""
        self._tool_descriptions = tool_registry.get_tool_descriptions()

    def process(
        self,
        query: str,
        user_id: str = "anonymous",
    ) -> ChatResponse:
        """处理用户查询并生成回答.

        Args:
            query: 用户查询文本
            user_id: 用户标识

        Returns:
            包含回答、来源、工具调用链等信息的响应
        """
        start_time = time.time()
        tool_calls: list[ToolCallInfo] = []

        # Step 1: 输入净化 - 检测注入攻击
        is_safe, _ = input_sanitizer.sanitize(query)
        if not is_safe:
            elapsed = (time.time() - start_time) * 1000
            return ChatResponse(
                answer="检测到潜在的提示词注入攻击，已拦截本次请求。"
                "请确保您的输入合规。",
                sources=[],
                tool_calls=[],
                processing_time_ms=round(elapsed, 2),
                injection_detected=True,
            )

        # Step 2: 调用LLM进行意图识别和工具选择
        _source_list = "\n".join(f"  - {s}" for s in sorted(_VALID_SOURCES))
        system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=self._tool_descriptions,
            available_sources=_source_list,
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        print(f"\n 第一轮LLM输入: {messages}")

        llm_response = llm_client.chat(
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        assistant_reply = llm_response["content"]
        
        print(f"\n 第一轮LLM输出: {llm_response}")
        
        # Step 3: 检查是否需要调用工具
        tool_call = _parse_tool_call(assistant_reply)
        
        print(f"\n tool_call:[{tool_call}]")
        
        max_loops = 3
        loop_count = 0

        while tool_call and loop_count < max_loops:
            loop_count += 1
            tool_name = tool_call.get("tool", "")
            arguments = tool_call.get("arguments", {})

            try:
                result = tool_registry.execute_tool(tool_name, **arguments)
                result_str = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                result_str = f"工具执行错误: {str(e)}"

            tool_calls.append(
                ToolCallInfo(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result_str[:500],
                )
            )

            # 将工具结果加入对话，让LLM生成最终回答
            messages.append(
                {"role": "assistant", "content": assistant_reply}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"工具 '{tool_name}' 的返回结果如下：\n"
                        f"{result_str}\n\n"
                        "请基于以上结果给出专业回答。\n"
                        "强制约束（必须遵守）：\n"
                        "1. 你只能引用上述返回结果中 source_tag 字段包含的来源标签\n"
                        "2. 引用格式为 [文档名:章节号]，例如 [信贷业务制度:4.1]\n"
                        "3. 严禁编造任何返回结果中不存在的来源\n"
                        "4. 如果返回结果为空或与问题不相关，"
                        "回复'根据现有知识库无法确认，请咨询相关部门'"
                    ),
                }
            )
            
            print(f"\n 第{loop_count + 1}轮LLM输入: {messages}")
            
            llm_response = llm_client.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
            )
            assistant_reply = llm_response["content"]
            
            print(f"\n 第{loop_count + 1}轮LLM输出: {assistant_reply}")
            
            # 检查是否还有新的工具调用
            tool_call = _parse_tool_call(assistant_reply)

        # Step 4: 过滤虚假来源（防幻觉）
        # 删除所有知识库中不存在的来源标注
        def _filter_fake_sources(text: str) -> tuple[str, list[str]]:
            valid: list[str] = []
            
            def _replace_fake(m: re.Match) -> str:
                full = m.group(0)
                inner = m.group(1).strip()
                # 去除 "来源文档:" 前缀
                inner = re.sub(r"^来源文档[:：\s]*", "", inner)
                tag = f"[{inner}]"
                if tag in _VALID_SOURCES:
                    valid.append(inner)
                    # 保留原始引用但统一格式
                    return f"[{inner}]"
                return ""
            
            # 匹配所有 [xxx:yyy] 格式的来源标注
            filtered = re.sub(r"\[([^\]]*?:[^\]]*?)\]", _replace_fake, text)
            return filtered, valid

        filtered_answer, valid_sources = _filter_fake_sources(assistant_reply)

        # Step 5: 输出脱敏
        final_answer = output_desensitizer.desensitize(filtered_answer)

        elapsed = (time.time() - start_time) * 1000

        return ChatResponse(
            answer=final_answer,
            sources=valid_sources,
            tool_calls=tool_calls,
            processing_time_ms=round(elapsed, 2),
            injection_detected=False,
        )


# 全局单例
agent = ComplianceAgent()
