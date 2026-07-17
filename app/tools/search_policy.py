"""制度检索工具.

根据查询关键词检索制度条款，返回Top-K相关结果。
"""

from __future__ import annotations

from app.knowledge_base.retriever import retriever
from app.models.schemas import SearchResult
from app.tools.registry import tool_registry


@tool_registry.register(
    name="search_policy",
    description="检索银行信贷制度和合规手册中的相关条款。"
    "当用户询问制度规定、流程、标准等制度类问题时使用。",
)
def search_policy(query: str) -> list[dict]:
    """检索制度条款.

    基于语义相似度检索知识库，返回最相关的Top-K条款。

    Args:
        query: 查询关键词或问题描述

    Returns:
        检索结果列表，每条包含规则内容、来源标签和相似度

    Examples:
        >>> results = search_policy("贷款审批流程")
        >>> len(results) > 0
        True
        >>> results[0]["source_tag"]
        '[信贷业务制度:4.1]'
    """
    results: list[SearchResult] = retriever.retrieve(query)

    output = []
    for result in results:
        output.append(
            {
                "content": result.rule.content,
                "source_tag": result.source_tag,
                "title": result.rule.title,
                "similarity": result.similarity,
            }
        )

    return output
