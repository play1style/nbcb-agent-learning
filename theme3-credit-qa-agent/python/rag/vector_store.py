"""RAG 组件：向量库（Vector Store）。

- 使用 NumPy 实现余弦相似度近邻检索（向量已归一化，点积即余弦）
- 索引与分片元数据持久化到 JSON 文件，启动时按知识库指纹校验缓存，
  指纹一致则直接加载，避免重复调用嵌入模型
- 生产环境可将本组件替换为 FAISS / Milvus / pgvector / Elasticsearch，
  上层检索接口保持不变
"""
import json
import os

import numpy as np


class VectorStore:
    def __init__(self, index_path):
        self.index_path = index_path
        self.chunks_meta = []     # 分片元数据（chunk_id/doc_title/section/text）
        self.matrix = None        # shape: (n_chunks, dim)，行向量已归一化
        self.signature = None     # 知识库内容指纹
        self.embedder_name = None

    # ---------- 构建与持久化 ----------

    def build(self, chunks, vectors, signature, embedder_name):
        """由分片与向量构建索引，并持久化到磁盘。"""
        self.chunks_meta = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "section": c.section,
                "text": c.text,
            }
            for c in chunks
        ]
        self.matrix = np.asarray(vectors, dtype=np.float32)
        self.signature = signature
        self.embedder_name = embedder_name
        self._save()

    def load_if_valid(self, signature, embedder_name):
        """指纹与嵌入模型一致时加载缓存索引，返回 True；否则返回 False。"""
        if not os.path.exists(self.index_path):
            return False
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("signature") != signature:
                return False
            if data.get("embedder") != embedder_name:
                return False
            self.chunks_meta = data["chunks"]
            self.matrix = np.asarray(data["vectors"], dtype=np.float32)
            self.signature = data["signature"]
            self.embedder_name = data["embedder"]
            return True
        except (OSError, ValueError, KeyError):
            return False

    def _save(self):
        payload = {
            "signature": self.signature,
            "embedder": self.embedder_name,
            "chunks": self.chunks_meta,
            "vectors": self.matrix.tolist(),
        }
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    # ---------- 检索 ----------

    def search(self, query_vector, top_k=3):
        """余弦相似度近邻检索，返回 [(chunk_meta, score), ...] 降序排列。"""
        if self.matrix is None or len(self.chunks_meta) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        scores = self.matrix @ q  # 向量已归一化，点积即余弦相似度
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks_meta[i], float(scores[i])) for i in order]

    def __len__(self):
        return len(self.chunks_meta)
