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

    三种 Embedding 模式按优先级尝试：
    1. sentence-transformers（本地缓存模型）
    2. 外部 Embedding API（通过 API 密钥调用，如智谱 embedding-3）
    3. 字符级 TF-IDF + 余弦相似度（兜底方案）
    """

    def __init__(
        self,
        policy_file: str,
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        *,
        config: Optional["Config"] = None,
    ):
        """
        Args:
            policy_file:      JSON 政策文件路径。
            embedding_model:  sentence-transformers 模型名称。
            config:           完整配置对象（用于提取外部 Embedding API 凭证）。
        """
        self.policy_file = policy_file
        self.embedding_model_name = embedding_model

        self.policies: List[Dict[str, Any]] = []
        self._embedder = None
        self._embeddings: Optional[np.ndarray] = None
        self._tfidf = None
        self._tfidf_matrix = None
        self._api_config = None
        self._api_embeddings: Optional[np.ndarray] = None

        # 如果传入了 Config 且有 API Key，则初始化外部 API Embedding 配置
        if config and getattr(config, "EMBEDDING_API_KEY", None):
            self._api_config = {
                "api_key": config.EMBEDDING_API_KEY,
                "api_base": config.EMBEDDING_API_BASE,
                "model": config.EMBEDDING_API_MODEL,
            }

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
        """三级降级：sentence-transformers → 外部 Embedding API → TF-IDF。"""
        # ── 优先：sentence-transformers 本地缓存 ──
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
            return
        except Exception as exc:
            logger.info("sentence-transformers unavailable: %s", exc)

        # ── 次选：外部 Embedding API ──
        if self._api_config is not None:
            try:
                self._init_api_embeddings()
                return
            except Exception as exc:
                logger.warning("External embedding API failed (%s); falling back to TF-IDF", exc)

        # ── 兜底：TF-IDF ──
        self._init_tfidf()

    def _init_tfidf(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=5000)
        except ImportError:
            logger.error("scikit-learn not installed – keyword matching fallback is unreliable")

    # ── 外部 Embedding API ──────────────────────────────────────────

    def _init_api_embeddings(self) -> None:
        """通过外部 API 预计算所有制度条款的 Embedding 向量并归一化。"""
        texts = [_fmt(p) for p in self.policies]
        logger.info(
            "Computing %d embeddings via external API (%s) …",
            len(texts), self._api_config["model"],
        )
        raw = self._call_embedding_api(texts)
        self._api_embeddings = np.array(raw, dtype=np.float32)
        # L2 归一化
        norms = np.linalg.norm(self._api_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._api_embeddings /= norms
        logger.info("External API embeddings ready – shape %s", self._api_embeddings.shape)

    def _call_embedding_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用外部 Embedding API（兼容 OpenAI / Zhipu 格式）。

        Args:
            texts: 待编码的文本列表（单元素列表也适用于单个查询）。

        Returns:
            每个文本对应的 embedding 向量列表。

        Raises:
            RuntimeError: API 调用失败或返回格式异常。
        """
        import requests

        url = self._api_config["api_base"].rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_config['api_key']}",
            "Content-Type": "application/json",
        }
        # 单条文本时 API 可能期望字符串而非列表
        payload = {
            "model": self._api_config["model"],
            "input": texts if len(texts) > 1 else texts[0],
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text[:200]}")

        body = resp.json()
        data = body.get("data", [])
        if not data:
            raise RuntimeError("Empty embedding data from API")

        # 按 index 排序保证顺序
        data.sort(key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in data]

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

        根据当前激活的 Embedding 模式（本地 / API / TF-IDF / 关键词）自动选择检索路径。
        """
        if not self.policies:
            return []

        if self._embedder is not None:
            return self._search_emb(query, top_k)
        if self._api_embeddings is not None:
            return self._search_api(query, top_k)
        if self._tfidf is not None:
            return self._search_tfidf(query, top_k)
        return self._search_keyword(query, top_k)

    def _search_emb(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_vec = self._embedder.encode([query], normalize_embeddings=True)
        scores = np.dot(q_vec, self._embeddings.T)[0]
        idxs = np.argsort(scores)[-top_k:][::-1]
        return [{**self.policies[i], "similarity": round(float(scores[i]), 4)} for i in idxs]

    def _search_api(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """通过外部 API 对查询编码，与预计算的文档向量做余弦相似度检索。"""
        q_vec = np.array(self._call_embedding_api([query]), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm
        scores = np.dot(q_vec, self._api_embeddings.T)[0]
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
