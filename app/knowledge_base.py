"""知识库模块——加载制度条款并提供基于 Embedding（或 TF-IDF）的语义检索能力。"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    内存知识库——对制度文档建立向量索引，返回与查询语义最相似的 Top-K 条。

    优先使用 sentence-transformers（多语言 MiniLM），若模型未缓存则自动
    降级为字符级 TF-IDF + 余弦相似度。
    """

    def __init__(
        self,
        policy_file: str,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Args:
            policy_file:      JSON 政策文件路径。
            embedding_model:  sentence-transformers 模型名称。
        """
        self.policy_file = policy_file
        self.embedding_model_name = embedding_model

        self.policies: List[Dict[str, Any]] = []
        self._embedder = None
        self._embeddings: Optional[np.ndarray] = None
        self._tfidf = None
        self._tfidf_matrix = None

        self._load_policies()
        self._init_embedder()
        self._index()

    # ── 初始化 ──────────────────────────────────────────────────────

    def _load_policies(self) -> None:
        path = Path(self.policy_file)
        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_file}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.policies = data.get("policies", [])
        logger.info("Loaded %d policy entries from %s", len(self.policies), self.policy_file)

    def _init_embedder(self) -> None:
        """尝试从缓存加载 sentence-transformers 模型；不可用时降级为 TF-IDF。"""
        try:
            from huggingface_hub import try_to_load_from_cache
            snapshot_dir = try_to_load_from_cache(
                f"sentence-transformers/{self.embedding_model_name}",
                filename="model.safetensors",
            )
            if snapshot_dir is None:
                raise RuntimeError("Model not cached – skip sentence-transformers")

            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                local_files_only=True,
            )
            logger.info("sentence-transformers ready (%s) [from cache]", self.embedding_model_name)
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s); falling back to TF-IDF", exc)
            self._init_tfidf()

    def _init_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=5000)
        except ImportError:
            logger.error("scikit-learn not installed – keyword matching fallback is unreliable")

    def _index(self) -> None:
        """为每条制度条款计算 Embedding 或 TF-IDF 向量。"""
        texts = [_fmt(p) for p in self.policies]

        if self._embedder is not None:
            self._embeddings = self._embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        elif self._tfidf is not None:
            self._tfidf_matrix = self._tfidf.fit_transform(texts)
            from sklearn.preprocessing import Normalizer
            self._tfidf_matrix = Normalizer(norm="l2").fit_transform(self._tfidf_matrix)

    # ── 语义检索 ────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        返回与 *query* 语义最相似的 *top_k* 条制度条款，按相似度降序排列。
        """
        if not self.policies:
            return []

        if self._embedder is not None:
            return self._search_emb(query, top_k)
        if self._tfidf is not None:
            return self._search_tfidf(query, top_k)
        return self._search_keyword(query, top_k)

    def _search_emb(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_vec = self._embedder.encode([query], normalize_embeddings=True)
        scores = np.dot(q_vec, self._embeddings.T)[0]
        idxs = np.argsort(scores)[-top_k:][::-1]
        return [{**self.policies[i], "similarity": round(float(scores[i]), 4)} for i in idxs]

    def _search_tfidf(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        from sklearn.preprocessing import Normalizer
        q_vec = Normalizer(norm="l2").fit_transform(self._tfidf.transform([query]))
        scores = (q_vec @ self._tfidf_matrix.T).toarray()[0:1].flatten()
        idxs = np.argsort(scores)[-top_k:][::-1]
        return [{**self.policies[i], "similarity": round(float(scores[i]), 4)} for i in idxs]

    def _search_keyword(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_words = set(query.lower().split())
        scored: List[tuple] = []
        for p in self.policies:
            text = _fmt(p).lower()
            hits = sum(1 for w in q_words if w in text)
            if hits:
                scored.append((hits / max(len(q_words), 1), p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**p, "similarity": round(s, 4)} for s, p in scored[:top_k]]


def _fmt(policy: Dict[str, Any]) -> str:
    """将一条制度条款拼接为可搜索的扁平字符串。"""
    parts = [policy.get("title", ""), policy.get("content", "")]
    kw = policy.get("keywords", [])
    if kw:
        parts.append(" ".join(kw))
    return " ".join(parts)
