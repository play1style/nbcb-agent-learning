"""RAG 组件：文档与分片数据结构。

Document：一份完整的知识库源文档（如一份产品说明书）。
Chunk：   文档切分后的最小检索单元，携带来源元数据，用于检索命中后的引用溯源。
"""
from dataclasses import dataclass


@dataclass
class Document:
    doc_id: str      # 文档唯一标识（文件名）
    title: str       # 文档标题（Markdown 一级标题）
    path: str        # 源文件路径
    content: str     # 文档全文


@dataclass
class Chunk:
    chunk_id: str    # 分片唯一标识（doc_id#序号）
    doc_id: str      # 所属文档
    doc_title: str   # 所属文档标题（用于引用展示）
    section: str     # 所属章节（如"申请材料"，用于引用展示与章节级路由）
    text: str        # 分片正文
