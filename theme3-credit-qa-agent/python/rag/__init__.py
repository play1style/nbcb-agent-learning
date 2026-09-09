"""RAG 组件包：文档加载、切分、嵌入、向量库、检索器。

各组件职责单一、接口清晰，可独立替换为生产级实现：
- loader        → 可替换为 PDF/Word/Confluence 加载器
- embeddings    → 可替换为 bge / m3e / OpenAI 兼容 Embedding API
- vector_store  → 可替换为 FAISS / Milvus / pgvector / Elasticsearch
- retriever     → 可扩展重排序（Rerank）、混合检索（Hybrid Search）
"""
