"""语义检索模块.

基于TF-IDF向量化和余弦相似度实现语义检索，返回Top-K相关条款。
"""

from __future__ import annotations

import math
from collections import Counter
from typing import ClassVar

from app.config import config
from app.knowledge_base.data import KnowledgeBase, knowledge_base as _default_kb
from app.models.schemas import KnowledgeRule, SearchResult


class TfIdfRetriever:
    """TF-IDF语义检索器.

    基于词频-逆文档频率（TF-IDF）和余弦相似度进行语义检索。
    无需外部嵌入API或深度学习模型，适合轻量级原型。

    Examples:
        >>> retriever = TfIdfRetriever()
        >>> results = retriever.retrieve("贷款审批流程")
        >>> len(results) > 0
        True
        >>> results[0].source_tag
        '[信贷业务制度:4.1]'
    """

    def __init__(
        self, knowledge_base: KnowledgeBase | None = None
    ) -> None:
        """初始化检索器.

        Args:
            knowledge_base: 知识库实例，默认使用全局单例
        """
        self._kb = knowledge_base or _default_kb
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """简单中文分词.

        基于字符二元组+标点分隔进行简单分词。

        Args:
            text: 待分词文本

        Returns:
            词项列表
        """
        # 替换常见分隔符
        for ch in "，。、；：？！""''（）【】《》\n\r\t":
            text = text.replace(ch, " ")

        tokens = []
        for part in text.split():
            if not part.strip():
                continue
            # 对每个部分提取字符双元组特征
            part = part.strip()
            if len(part) <= 2:
                tokens.append(part)
            else:
                # 加入单字符
                for ch in part:
                    tokens.append(ch)
                # 加入双元组
                for i in range(len(part) - 1):
                    tokens.append(part[i : i + 2])
                # 加入三元组
                for i in range(len(part) - 2):
                    tokens.append(part[i : i + 3])
        return tokens

    def _build_index(self) -> None:
        """构建TF-IDF索引。"""
        rules = self._kb.rules
        self._doc_count = len(rules)
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter] = []

        # 文档频率（包含词项的文档数）
        df: Counter[str] = Counter()

        for rule in rules:
            tokens = self._tokenize(rule.content + " " + rule.title)
            self._doc_tokens.append(tokens)
            token_counter = Counter(tokens)
            self._doc_freqs.append(token_counter)
            for token in set(tokens):
                df[token] += 1

        self._df = df
        self._idf: dict[str, float] = {}
        for token, doc_freq in df.items():
            self._idf[token] = math.log(
                (self._doc_count + 1) / (doc_freq + 1) + 1
            )

    def _compute_tfidf_vector(
        self, tokens: list[str]
    ) -> dict[str, float]:
        """计算文本的TF-IDF向量.

        Args:
            tokens: 词项列表

        Returns:
            词项到TF-IDF值的映射
        """
        if not tokens:
            return {}
        counter = Counter(tokens)
        max_freq = max(counter.values())
        vector: dict[str, float] = {}
        for token, freq in counter.items():
            if token in self._idf:
                tf = freq / max_freq
                vector[token] = tf * self._idf[token]
        return vector

    def _cosine_similarity(
        self, vec1: dict[str, float], vec2: dict[str, float]
    ) -> float:
        """计算两个向量的余弦相似度.

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度 (0-1)
        """
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0

        for token, val in vec1.items():
            norm1 += val * val
            if token in vec2:
                dot += val * vec2[token]

        for val in vec2.values():
            norm2 += val * val

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """根据查询语句检索最相关的制度条款.

        Args:
            query: 查询语句
            top_k: 返回结果数量，默认使用配置值

        Returns:
            按相似度降序排列的检索结果列表
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf_vector(query_tokens)

        scores: list[tuple[float, int]] = []
        for idx, doc_tokens in enumerate(self._doc_tokens):
            doc_vector = self._compute_tfidf_vector(doc_tokens)
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scores.append((similarity, idx))

        # 按相似度降序排列
        scores.sort(key=lambda x: x[0], reverse=True)

        results: list[SearchResult] = []
        for similarity, idx in scores:
            if similarity < config.RETRIEVAL_SIMILARITY_THRESHOLD:
                continue
            if len(results) >= top_k:
                break
            rule = self._kb.rules[idx]
            source_tag = f"[{rule.document}:{rule.section}]"
            results.append(
                SearchResult(
                    rule=rule,
                    similarity=round(similarity, 4),
                    source_tag=source_tag,
                )
            )

        return results


# 全局单例
retriever = TfIdfRetriever()
