"""工具注册中心.

提供统一的工具注册和调用接口，便于Agent自动发现和调用工具。
"""

from __future__ import annotations

from typing import Any, Callable


class Tool:
    """工具描述类.

    Attributes:
        name: 工具名称
        description: 工具描述
        func: 工具函数
        parameters: 参数模式描述
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., Any],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """初始化工具.

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数
            parameters: 参数模式描述
        """
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or {}

    def execute(self, **kwargs: Any) -> Any:
        """执行工具函数.

        Args:
            **kwargs: 工具参数

        Returns:
            工具执行结果
        """
        return self.func(**kwargs)

    def to_prompt_description(self) -> str:
        """生成工具描述的提示文本.

        Returns:
            工具描述的格式化字符串
        """
        params_desc = []
        for name, param_info in self.parameters.items():
            required = "（必填）" if param_info.get("required") else "（可选）"
            desc = param_info.get("description", "")
            params_desc.append(f"    - {name}: {desc} {required}")

        param_str = "\n".join(params_desc) if params_desc else "    无参数"
        return (
            f"工具名称: {self.name}\n"
            f"描述: {self.description}\n"
            f"参数:\n{param_str}"
        )


# 内置工具注册信息，便于Agent系统提示生成
TOOL_METADATA: dict[str, dict[str, Any]] = {
    "search_policy": {
        "name": "search_policy",
        "description": "检索银行信贷制度和合规手册中的相关条款。当用户询问制度规定、流程、标准等制度类问题时使用。",
        "parameters": {
            "query": {
                "type": "string",
                "description": "查询关键词或问题描述",
                "required": True,
            }
        },
    },
    "calculate_loan": {
        "name": "calculate_loan",
        "description": "计算等额本息贷款的月供和总利息。当用户询问贷款金额计算时使用。",
        "parameters": {
            "principal": {
                "type": "number",
                "description": "贷款本金（万元）",
                "required": True,
            },
            "annual_rate": {
                "type": "number",
                "description": "年利率（百分比，如3.5表示3.5%%）",
                "required": True,
            },
            "years": {
                "type": "integer",
                "description": "贷款期限（年）",
                "required": True,
            },
        },
    },
}


class ToolRegistry:
    """工具注册中心.

    管理所有可用工具的注册、查询和执行。
    工具需通过装饰器或register方法注册。

    Examples:
        >>> registry = ToolRegistry()
        >>> @registry.register("hello", "打招呼", lambda name: f"Hello {name}")
        ... def greet(name: str) -> str:
        ...     return f"Hello {name}"
        >>> registry.list_tools()
        ['hello']
    """

    def __init__(self) -> None:
        """初始化工具注册中心。"""
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable:
        """注册工具（装饰器模式）.

        Args:
            name: 工具名称
            description: 工具描述
            parameters: 参数模式描述

        Returns:
            装饰器函数
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_params = parameters
            if tool_params is None and name in TOOL_METADATA:
                tool_params = TOOL_METADATA[name].get("parameters")
            self._tools[name] = Tool(
                name=name,
                description=description,
                func=func,
                parameters=tool_params,
            )
            return func

        return decorator

    def get_tool(self, name: str) -> Tool | None:
        """根据名称获取工具.

        Args:
            name: 工具名称

        Returns:
            工具实例，不存在则返回None
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有已注册的工具名称.

        Returns:
            工具名称列表
        """
        return list(self._tools.keys())

    def execute_tool(
        self, name: str, **kwargs: Any
    ) -> Any:
        """执行指定工具.

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果

        Raises:
            ValueError: 工具不存在
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"工具 '{name}' 未注册")
        return tool.execute(**kwargs)

    def get_tool_descriptions(self) -> str:
        """获取所有工具的描述文本，供Agent系统提示使用.

        Returns:
            工具描述文本
        """
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(tool.to_prompt_description())
        return "\n\n".join(descriptions)


# 全局单例
tool_registry = ToolRegistry()
