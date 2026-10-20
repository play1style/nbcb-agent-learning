"""Markdown 知识加载、章节分块、向量索引和语义检索。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .embeddings import (
    EmbeddingClient,
    cosine_similarity,
    lexical_overlap,
)
from .models import SourceReference


HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
# 只把 1、1.1、1.1.1 等编号章节写为来源章节号。
SECTION_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)")


@dataclass(slots=True)
class KnowledgeChunk:
    """可直接用于答案引用的 Markdown 制度章节分块。"""

    chunk_id: str
    document: str
    section: str
    title: str
    text: str
    source_path: str

    @property
    def citation(self) -> str:
        """返回 ``[来源文档:章节号]`` 格式的引用。"""
        return f"[{self.document}:{self.section}]"


@dataclass(slots=True)
class IndexedChunk:
    """带持久化 Embedding 向量的知识分块。"""

    chunk: KnowledgeChunk
    vector: list[float]
    score: float = 0.0


def load_markdown_chunks(knowledge_dir: Path) -> list[KnowledgeChunk]:
    """加载全部 Markdown 文件，并按编号标题切分为章节。"""
    chunks: list[KnowledgeChunk] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(_parse_markdown(path, text))
    return chunks


def _parse_markdown(path: Path, text: str) -> list[KnowledgeChunk]:
    """将单个 Markdown 文件解析为多个章节分块。"""
    lines = text.splitlines()
    document_title = path.stem
    current_section = "0"
    current_title = "文档概述"
    buffer: list[str] = []
    chunks: list[KnowledgeChunk] = []

    def flush() -> None:
        """遇到下一个标题时，将当前缓冲区保存为一个章节分块。"""
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if not body:
            buffer = []
            return
        # 分块 ID 基于文件、章节和正文生成，便于失败后稳定重建。
        chunk_key = f"{path.name}:{current_section}:{current_title}:{body}"
        chunk_id = hashlib.sha256(chunk_key.encode("utf-8")).hexdigest()[:20]
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                document=document_title,
                section=current_section,
                title=current_title,
                text=body,
                source_path=str(path),
            )
        )
        buffer = []

    for raw_line in lines:
        heading_match = HEADING_PATTERN.match(raw_line)
        if not heading_match:
            buffer.append(raw_line)
            continue

        level = len(heading_match.group(1))
        heading_text = heading_match.group(2).strip()
        if level == 1:
            document_title = heading_text
            continue

        flush()
        section_match = SECTION_PATTERN.match(heading_text)
        if section_match:
            current_section = section_match.group(1)
            current_title = heading_text
        else:
            current_section = _slug_section(heading_text)
            current_title = heading_text

    flush()
    return chunks


def _slug_section(title: str) -> str:
    """为没有数字编号的标题生成稳定章节标识。"""
    ascii_title = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", title).strip("-")
    return ascii_title[:40] or "section"


def fingerprint_documents(knowledge_dir: Path) -> str:
    """计算知识文件指纹，用于判断是否需要重建索引。"""
    digest = hashlib.sha256()
    for path in sorted(knowledge_dir.glob("*.md")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class VectorStore:
    """持久化并查询 Markdown 制度分块的 JSON 向量索引。"""

    VERSION = 1

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingClient,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.path = settings.vector_store_path
        self.entries: list[IndexedChunk] = []
        self.metadata: dict = {}

    def ensure_index(self, force: bool = False) -> bool:
        """加载有效索引；索引缺失或过期时自动重建。"""
        expected_fingerprint = fingerprint_documents(self.settings.knowledge_dir)
        if not force and self._load_valid(expected_fingerprint):
            return False
        self.build(force=force)
        return True

    def build(self, force: bool = False) -> None:
        """构建 Embedding 索引并写入 ``vector_store.json``。"""
        document_fingerprint = fingerprint_documents(self.settings.knowledge_dir)
        if (
            not force
            and self._load_valid(document_fingerprint)
            and self.metadata.get("embedding_backend") == self.embedder.identity
        ):
            return

        knowledge_chunks = load_markdown_chunks(self.settings.knowledge_dir)
        if not knowledge_chunks:
            raise RuntimeError(
                f"知识库目录中没有 Markdown 文档: {self.settings.knowledge_dir}"
            )
        vectors = self.embedder.embed([chunk.text for chunk in knowledge_chunks])
        self.entries = [
            IndexedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(knowledge_chunks, vectors)
        ]
        self.metadata = {
            "version": self.VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "document_fingerprint": document_fingerprint,
            "embedding_backend": self.embedder.identity,
            "dimension": len(vectors[0]) if vectors else 0,
            "document_count": len({chunk.document for chunk in knowledge_chunks}),
            "chunk_count": len(knowledge_chunks),
        }
        self._persist()

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[IndexedChunk]:
        """使用语义相似度和关键词重合度混合排序，返回 Top-K 分块。"""
        if not self.entries:
            self.ensure_index()
        if not query.strip():
            return []
        query_vector = self.embedder.embed([query])[0]
        top_k = top_k or self.settings.retrieval_top_k
        scored: list[IndexedChunk] = []
        for entry in self.entries:
            semantic = max(
                0.0,
                cosine_similarity(query_vector, entry.vector),
            )
            lexical = lexical_overlap(
                query,
                f"{entry.chunk.title}\n{entry.chunk.text}",
            )
            title_match = lexical_overlap(query, entry.chunk.title)
            # 语义向量权重 35%，正文词元重合 50%，标题命中 15%。
            score = (
                (0.35 * semantic)
                + (0.50 * lexical)
                + (0.15 * title_match)
            )
            scored.append(
                IndexedChunk(
                    chunk=entry.chunk,
                    vector=entry.vector,
                    score=round(max(0.0, min(1.0, score)), 6),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, top_k)]

    def source_references(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SourceReference]:
        """将 Top-K 检索结果转换为 API 可序列化的来源引用。"""
        return [
            SourceReference(
                document=item.chunk.document,
                section=item.chunk.section,
                content=item.chunk.text,
                score=item.score,
            )
            for item in self.search(query, top_k=top_k)
        ]

    def _load_valid(self, expected_fingerprint: str) -> bool:
        """仅在索引版本、知识指纹和 Embedding 模型全部匹配时加载。"""
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            if metadata.get("version") != self.VERSION:
                return False
            if metadata.get("document_fingerprint") != expected_fingerprint:
                return False
            if metadata.get("embedding_backend") != self.embedder.identity:
                return False
            entries = []
            for item in payload.get("entries", []):
                chunk = KnowledgeChunk(**item["chunk"])
                entries.append(
                    IndexedChunk(
                        chunk=chunk,
                        vector=[float(value) for value in item["vector"]],
                    )
                )
            self.entries = entries
            self.metadata = metadata
            return bool(entries)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _persist(self) -> None:
        """通过临时文件加原子替换，避免索引写入中断后损坏。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "entries": [
                {
                    "chunk": asdict(entry.chunk),
                    "vector": entry.vector,
                }
                for entry in self.entries
            ],
        }
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)


class PolicyRetriever:
    """供 Agent 工具调用的制度检索服务。"""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[IndexedChunk]:
        """检索 Top-K 制度分块。"""
        return self.vector_store.search(query=query, top_k=top_k)

    def retrieve_sources(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SourceReference]:
        """检索 Top-K 制度分块并转换为来源引用。"""
        return self.vector_store.source_references(query=query, top_k=top_k)
