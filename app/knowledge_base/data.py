"""知识库数据管理模块.

负责加载、管理和查询知识库中的制度规则数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

from app.config import config
from app.models.schemas import KnowledgeRule


class KnowledgeBase:
    """知识库管理器.

    从JSON文件加载知识库数据，提供规则查询和管理功能。

    Examples:
        >>> kb = KnowledgeBase()
        >>> len(kb.rules) > 0
        True
        >>> rule = kb.get_rule(1)
        >>> rule.title == "贷款审批流程"
        True
    """

    def __init__(
        self, data_path: str | None = None
    ) -> None:
        """初始化知识库管理器.

        Args:
            data_path: 知识库JSON文件路径，默认使用配置中的路径

        Raises:
            FileNotFoundError: 知识库文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        self._data_path = Path(data_path or config.KNOWLEDGE_BASE_PATH)
        self._rules: list[KnowledgeRule] = []
        self._rule_map: dict[int, KnowledgeRule] = {}
        self._load()

    def _load(self) -> None:
        """从JSON文件加载知识库数据。"""
        if not self._data_path.exists():
            raise FileNotFoundError(
                f"知识库文件不存在: {self._data_path}"
            )
        with open(self._data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._rules = [KnowledgeRule(**rule) for rule in data["rules"]]
        self._rule_map = {rule.id: rule for rule in self._rules}

    @property
    def rules(self) -> list[KnowledgeRule]:
        """获取所有规则条目."""
        return list(self._rules)

    @property
    def rule_count(self) -> int:
        """获取规则总数."""
        return len(self._rules)

    def get_rule(self, rule_id: int) -> KnowledgeRule | None:
        """根据ID获取规则条目.

        Args:
            rule_id: 规则ID

        Returns:
            匹配的规则，不存在则返回None
        """
        return self._rule_map.get(rule_id)

    def search_by_keyword(self, keyword: str) -> list[KnowledgeRule]:
        """根据关键词精确匹配搜索规则.

        简单的关键词匹配搜索，用于语义搜索的补充。

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的规则列表
        """
        keyword_lower = keyword.lower()
        results = []
        for rule in self._rules:
            if (
                keyword_lower in rule.content.lower()
                or keyword_lower in rule.title.lower()
            ):
                results.append(rule)
        return results


# 全局单例
knowledge_base = KnowledgeBase()
