"""RAG 知识库模块（知识中枢）。

将 RAG 各组件装配为一条完整的检索增强管线：

    知识库目录（knowledge/*.md）
        → DocumentLoader  文档加载
        → TextSplitter    章节级切分
        → Embedder        文本嵌入（sentence-transformers / Embedding API）
        → VectorStore     向量索引构建与持久化（指纹校验缓存）
        → Retriever       Top-K 语义检索（无嵌入后端时降级 BM25 稀疏检索）

扩展知识库只需向 knowledge/ 目录添加 Markdown 文档并重启（或重建索引），
无需修改任何检索与问答代码——这正是 RAG 架构相对硬编码知识库的核心优势。
"""
import os
from pathlib import Path

from rag.embeddings import create_embedder
from rag.loader import knowledge_signature, load_documents, split_documents
from rag.retriever import Retriever
from rag.vector_store import VectorStore

# 知识库目录：默认取案例根目录下的 knowledge/，Python 与 Java 双版本共用
DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent / "vector_store.json"


class RAGKnowledgeBase:
    """基于 RAG 组件的产品知识库。"""

    def __init__(self, knowledge_dir=None, index_path=None, min_score=0.35):
        self.knowledge_dir = Path(knowledge_dir or DEFAULT_KNOWLEDGE_DIR)
        self.index_path = str(index_path or DEFAULT_INDEX_PATH)
        self.min_score = float(os.environ.get("RAG_MIN_SCORE", min_score))

        # 1. 文档加载与切分
        self.documents = load_documents(self.knowledge_dir)
        self.chunks = split_documents(self.documents)

        # 2. 嵌入组件（不可用时为 None，降级 BM25）
        self.embedder = create_embedder()

        # 3. 向量索引：指纹一致直接加载缓存，否则重新嵌入并持久化
        self.vector_store = None
        if self.embedder:
            self.vector_store = VectorStore(self.index_path)
            signature = knowledge_signature(self.knowledge_dir)
            if self.vector_store.load_if_valid(signature, self.embedder.name):
                print(f"[RAG] 向量索引命中缓存：{self.index_path}"
                      f"（{len(self.vector_store)} 个分片）")
            else:
                texts = [self._chunk_text(c) for c in self.chunks]
                vectors = self.embedder.embed(texts)
                self.vector_store.build(
                    self.chunks, vectors, signature, self.embedder.name
                )
                print(f"[RAG] 向量索引已重建并持久化：{self.index_path}"
                      f"（{len(self.vector_store)} 个分片）")

        # 4. 检索器
        self.retriever = Retriever(
            self.chunks,
            embedder=self.embedder,
            vector_store=self.vector_store,
            min_score=self.min_score,
        )
        print(f"[RAG] 知识库就绪：{len(self.documents)} 份文档 / "
              f"{len(self.chunks)} 个分片 / 检索模式 {self.retriever.mode}")

    @staticmethod
    def _chunk_text(chunk):
        """嵌入文本 = 标题 + 章节 + 正文，提升检索相关性。"""
        return f"{chunk.doc_title}｜{chunk.section}｜{chunk.text}"

    def retrieve(self, query, top_k=3):
        """语义检索与 query 最相关的知识分片（RetrievalResult 列表）。"""
        return self.retriever.retrieve(query, top_k=top_k)

    def stats(self):
        """知识库运行状态，用于持久化备份与审计。"""
        return {
            "knowledge_dir": str(self.knowledge_dir),
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "documents": [d.doc_id for d in self.documents],
            "embedding": self.embedder.name if self.embedder else "无（BM25 降级）",
            "retrieval_mode": self.retriever.mode,
            "min_score": self.min_score,
            "index_path": self.index_path if self.vector_store else None,
        }
