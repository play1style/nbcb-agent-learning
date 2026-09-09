"""RAG 组件：检索器（Retriever）。

对上层（Agent）提供统一的语义检索接口：
- 稠密检索：查询嵌入 + 向量库近邻搜索 + 相似度阈值过滤
- 稀疏检索：无嵌入后端时降级为 BM25
- 每次检索返回带来源与得分的 RetrievalResult，支撑引用溯源与审计
"""
from dataclasses import dataclass

from .bm25 import BM25Index
from .document import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk     # 命中的知识分片
    score: float     # 相似度/BM25 得分

    @property
    def source(self):
        """引用来源描述，如《小微快贷》·申请材料。"""
        return f"《{self.chunk.doc_title}》·{self.chunk.section}"


class Retriever:
    def __init__(self, chunks, embedder=None, vector_store=None,
                 min_score=0.35):
        self.chunks = chunks
        self.embedder = embedder
        self.vector_store = vector_store
        self.min_score = min_score
        self.bm25 = None if embedder else BM25Index().build(chunks)

    @property
    def mode(self):
        return "dense" if self.embedder else "sparse(BM25)"

    def retrieve(self, query, top_k=3):
        """检索与 query 最相关的知识分片。"""
        if self.embedder:
            return self._dense_retrieve(query, top_k)
        return self._sparse_retrieve(query, top_k)

    def _dense_retrieve(self, query, top_k):
        qvec = self.embedder.embed([query])[0]
        hits = self.vector_store.search(qvec, top_k=top_k)
        results = []
        for meta, score in hits:
            if score < self.min_score:
                continue
            results.append(RetrievalResult(chunk=_meta_to_chunk(meta), score=score))
        return results

    def _sparse_retrieve(self, query, top_k):
        return [
            RetrievalResult(chunk=chunk, score=score)
            for chunk, score in self.bm25.search(query, top_k=top_k)
        ]


def _meta_to_chunk(meta):
    return Chunk(
        chunk_id=meta["chunk_id"],
        doc_id=meta["doc_id"],
        doc_title=meta["doc_title"],
        section=meta["section"],
        text=meta["text"],
    )
