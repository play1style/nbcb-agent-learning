"""RAG 组件：文本嵌入（Embedding）。

将文本映射为稠密向量，是语义检索的基础。支持两类真实嵌入后端，按优先级自动选择：

1. 本地模型（sentence-transformers，默认 BAAI/bge-small-zh-v1.5）
   - 首次运行自动下载模型，之后完全离线可用，满足银行内网要求
2. OpenAI 兼容 Embedding API（/v1/embeddings）
   - 通过环境变量配置，适配行内统一部署的 Embedding 服务

若两类后端均不可用，create_embedder() 返回 None，
由调用方降级为 BM25 稀疏检索（见 bm25.py），保证演示始终可运行。

环境变量（均为可选）：
  EMBEDDING_API_KEY   – Embedding API 密钥（缺省回退读取 LLM_API_KEY）
  EMBEDDING_BASE_URL  – Embedding API 基地址（缺省回退读取 LLM_BASE_URL）
  EMBEDDING_MODEL     – 嵌入模型名称
"""
import json
import os
import socket
import urllib.error
import urllib.request


class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的本地嵌入模型（离线可用）。"""

    backend = "sentence-transformers"

    def __init__(self, model_name=None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )
        self.model = SentenceTransformer(self.model_name)

    @property
    def name(self):
        return f"{self.backend}:{self.model_name}"

    def embed(self, texts):
        """texts: List[str] -> List[List[float]]（已做 L2 归一化）"""
        vectors = self.model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class ApiEmbedder:
    """基于 OpenAI 兼容接口（POST {base_url}/embeddings）的嵌入服务。"""

    backend = "embedding-api"

    def __init__(self):
        self.api_key = (
            os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("LLM_API_KEY", "")
        ).strip()
        self.base_url = (
            os.environ.get("EMBEDDING_BASE_URL")
            or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.model_name = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.timeout = int(os.environ.get("EMBEDDING_TIMEOUT", "15"))

    @property
    def name(self):
        return f"{self.backend}:{self.model_name}"

    def embed(self, texts):
        """texts: List[str] -> List[List[float]]（已做 L2 归一化）"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model_name, "input": list(texts)}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        vectors = [None] * len(result["data"])
        for item in result["data"]:
            vectors[item["index"]] = _normalize(item["embedding"])
        return vectors


def create_embedder():
    """按优先级创建嵌入组件；均不可用时返回 None（由上层降级为 BM25）。"""
    # 1. 本地 sentence-transformers 模型
    try:
        import sentence_transformers  # noqa: F401

        embedder = SentenceTransformerEmbedder()
        embedder.embed(["探测"])  # 探测模型可用性（含模型下载）
        print(f"[RAG] 嵌入组件：{embedder.name}")
        return embedder
    except Exception as e:
        print(f"[RAG] 本地嵌入模型不可用（{type(e).__name__}），尝试 Embedding API…")

    # 2. OpenAI 兼容 Embedding API
    api_key = os.environ.get("EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY", "")
    if api_key.strip():
        try:
            embedder = ApiEmbedder()
            embedder.embed(["探测"])  # 探测服务可用性
            print(f"[RAG] 嵌入组件：{embedder.name}")
            return embedder
        except (urllib.error.URLError, socket.timeout, ValueError, KeyError) as e:
            print(f"[RAG] Embedding API 不可用（{e}）")

    # 3. 降级：BM25 稀疏检索（由 knowledge_base 装配）
    print("[RAG] 无可用嵌入后端，降级为 BM25 稀疏检索（生产环境常用混合检索组件）")
    return None


def _normalize(vec):
    """L2 归一化，使余弦相似度可直接用点积计算。"""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]
