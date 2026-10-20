"""Embedding 客户端：支持智谱在线向量和离线确定性哈希向量。"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable

import httpx

from .config import Settings


logger = logging.getLogger(__name__)


def normalize_vector(vector: list[float]) -> list[float]:
    """对向量执行 L2 归一化，便于直接使用点积计算余弦相似度。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def tokenize(text: str) -> list[str]:
    """将中英文文本切分为字符、二元词和三元词，供检索计分使用。"""
    normalized = text.lower()
    ascii_tokens = re.findall(r"[a-z0-9]+", normalized)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", normalized)
    chinese_tokens: list[str] = []
    for run in chinese_runs:
        # 中文同时保留单字、双字和三字组合，兼顾短查询与专业术语。
        chinese_tokens.extend(run)
        chinese_tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
        chinese_tokens.extend(run[index : index + 3] for index in range(len(run) - 2))
    return ascii_tokens + chinese_tokens


class EmbeddingClient(ABC):
    """统一的 Embedding 客户端抽象接口。"""

    backend: str
    model_name: str

    @property
    def identity(self) -> str:
        """返回写入向量索引的稳定模型标识。"""
        return f"{self.backend}:{self.model_name}"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """将文本列表转换为归一化稠密向量。"""


class ZhipuEmbeddingClient(EmbeddingClient):
    """调用智谱 OpenAI 兼容格式的 Embedding 接口。"""

    backend = "zhipu"

    def __init__(self, settings: Settings) -> None:
        if not settings.zhipu_api_key:
            raise ValueError("ZHIPU_API_KEY 未配置")
        self.api_key = settings.zhipu_api_key
        self.url = settings.zhipu_embedding_url
        self.model_name = settings.embedding_model
        self.timeout = settings.embedding_timeout_seconds
        self.batch_size = max(1, settings.embedding_batch_size)
        self._client = httpx.Client(timeout=self.timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """分批生成向量，并校验接口返回数量和顺序。"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._request_with_retry(batch)
            items = sorted(response["data"], key=lambda item: item["index"])
            if len(items) != len(batch):
                raise RuntimeError("智谱 Embedding 返回的向量数量不匹配")
            vectors.extend(
                normalize_vector([float(value) for value in item["embedding"]])
                for item in items
            )
        return vectors

    def _request_with_retry(self, batch: list[str]) -> dict:
        """发送单批 Embedding 请求，对瞬时失败执行指数退避重试。"""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = self._client.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model_name, "input": batch},
                )
                response.raise_for_status()
                payload = response.json()
                if "data" not in payload:
                    raise RuntimeError(f"Embedding 响应缺少 data: {payload}")
                return payload
            except (httpx.HTTPError, KeyError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < 2:
                    # 0.5 秒、1 秒、2 秒的退避节奏可降低瞬时故障影响。
                    time.sleep(0.5 * (2**attempt))
        raise RuntimeError(f"智谱 Embedding 调用失败: {last_error}") from last_error


class HashEmbeddingClient(EmbeddingClient):
    """用于离线测试和演示的确定性特征哈希向量。"""

    backend = "hash"

    def __init__(self, dimension: int = 768) -> None:
        self.dimension = max(64, dimension)
        self.model_name = f"feature-hash-{self.dimension}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """在无网络环境下生成稳定、可复现的稠密向量。"""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            # 使用 BLAKE2 稳定散列到固定维度，并用最低位决定符号。
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dimension
            sign = 1.0 if raw & 1 else -1.0
            vector[index] += sign
        return normalize_vector(vector)


def create_embedder(
    settings: Settings,
    allow_fallback: bool | None = None,
) -> EmbeddingClient:
    """根据配置创建 Embedding 客户端，并支持可选的离线降级。"""
    provider = settings.embedding_provider.strip().lower()
    if provider == "hash":
        return HashEmbeddingClient(settings.embedding_dimension)
    if provider == "zhipu":
        return ZhipuEmbeddingClient(settings)
    if provider != "auto":
        raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {provider}")

    should_fallback = True if allow_fallback is None else allow_fallback
    if settings.zhipu_api_key:
        try:
            client = ZhipuEmbeddingClient(settings)
            # auto 模式启动时先真实探测一次，避免仅因存在密钥就误判可用。
            client.embed(["embedding service probe"])
            logger.info("Embedding backend selected: %s", client.identity)
            return client
        except (ValueError, RuntimeError) as exc:
            logger.warning("智谱 Embedding 不可用，切换离线向量: %s", exc)
            if not should_fallback:
                raise
    if should_fallback:
        return HashEmbeddingClient(settings.embedding_dimension)
    raise ValueError("未配置可用的 Embedding 模型")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个已归一化向量的余弦相似度。"""
    if len(left) != len(right):
        raise ValueError("向量维度不一致")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))


def lexical_overlap(query: str, content: str) -> float:
    """计算查询词与文档内容的词元重合比例。"""
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(tokenize(content))
    return len(query_tokens & content_tokens) / len(query_tokens)


def chunks(items: Iterable, size: int) -> Iterable[list]:
    """按固定大小切分可迭代对象。"""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
