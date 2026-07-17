"""Deepseek LLM客户端模块.

封装Deepseek API调用，提供聊天和工具调用接口。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import config


class LLMClient:
    """Deepseek LLM客户端.

    封装与Deepseek API的通信，支持对话、工具调用等功能。

    Examples:
        >>> client = LLMClient()
        >>> response = client.chat(
        ...     messages=[{"role": "user", "content": "你好"}]
        ... )
        >>> "content" in response
        True
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> None:
        """初始化LLM客户端.

        Args:
            api_key: Deepseek API密钥
            api_base: API基础地址
            model: 模型名称
        """
        self._api_key = api_key or config.DEEPSEEK_API_KEY
        self._api_base = api_base or config.DEEPSEEK_API_BASE
        self._model = model or config.DEEPSEEK_MODEL
        self._chat_url = f"{self._api_base}/v1/chat/completions"

        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """发送聊天请求到Deepseek API.

        Args:
            messages: 消息列表，每项包含role和content字段
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            timeout: 超时时间（秒）

        Returns:
            API响应字典，包含content、usage等信息

        Raises:
            httpx.HTTPError: API调用失败
        """
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                self._chat_url, headers=self._headers, json=payload
            )
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        usage = data.get("usage", {})

        return {
            "content": content or "",
            "finish_reason": choice["finish_reason"],
            "usage": usage,
        }

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tool_descriptions: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """带工具描述的聊天请求.

        在系统消息中注入工具描述，引导模型按JSON格式返回工具调用。

        Args:
            messages: 消息列表
            tool_descriptions: 工具描述文本
            temperature: 温度参数
            max_tokens: 最大生成token数
            timeout: 超时时间（秒）

        Returns:
            API响应字典
        """
        # 在最后一条用户消息后附加工具调用说明
        tool_call_instruction = (
            "\n\n【工具调用】\n"
            "如果需要使用工具获取信息，请在回答中使用以下JSON格式表示工具调用：\n"
            '{"tool": "工具名称", "arguments": {"参数名": "参数值"}}\n\n'
            f"可用工具：\n{tool_descriptions}\n\n"
            "如果不需要调用工具，直接回答即可。"
        )

        modified_messages = list(messages)
        if modified_messages and modified_messages[-1]["role"] == "user":
            modified_messages[-1] = {
                "role": "user",
                "content": modified_messages[-1]["content"]
                + tool_call_instruction,
            }

        return self.chat(
            messages=modified_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


# 全局单例
llm_client = LLMClient()
