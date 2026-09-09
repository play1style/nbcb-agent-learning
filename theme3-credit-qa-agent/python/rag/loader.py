"""RAG 组件：文档加载与切分（Document Loader + Text Splitter）。

- 从知识库目录加载全部 Markdown 源文档
- 按二级标题（## 章节）切分为检索分片；超长章节按段落二次切分
- 提供知识库内容指纹，用于向量索引的缓存失效判断
"""
import hashlib
import re
from pathlib import Path

from .document import Chunk, Document


def load_documents(knowledge_dir):
    """加载目录下全部 .md 文档，返回 Document 列表。"""
    knowledge_dir = Path(knowledge_dir)
    if not knowledge_dir.is_dir():
        raise FileNotFoundError(f"知识库目录不存在：{knowledge_dir}")

    documents = []
    for path in sorted(knowledge_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        m = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        title = m.group(1).strip() if m else path.stem
        documents.append(Document(
            doc_id=path.stem,
            title=title,
            path=str(path),
            content=content,
        ))
    if not documents:
        raise ValueError(f"知识库目录 {knowledge_dir} 下没有可用的 .md 文档")
    return documents


def split_document(doc, max_chars=500):
    """将一份文档切分为 Chunk 列表。

    切分策略：优先按 Markdown 二级标题（##）切分，保证每个分片语义完整；
    单个章节超过 max_chars 时，再按空行段落二次切分。
    """
    chunks = []
    sections = re.split(r"(?m)^##\s+", doc.content)

    # sections[0] 为一级标题与引言部分，信息量低时跳过
    pending = []
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue
        if i == 0:
            # 文档开头（含 # 标题），作为"概述"分片保留
            text = re.sub(r"(?m)^#\s+.+$", "", sec).strip()
            if text:
                pending.append(("概述", text))
            continue
        lines = sec.split("\n", 1)
        section_name = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if not body:
            continue
        if len(body) <= max_chars:
            pending.append((section_name, body))
        else:
            for j, para in enumerate(re.split(r"\n\s*\n", body)):
                para = para.strip()
                if para:
                    suffix = f"（{j + 1}）" if j > 0 else ""
                    pending.append((section_name + suffix, para))

    for idx, (section, text) in enumerate(pending):
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}#{idx}",
            doc_id=doc.doc_id,
            doc_title=doc.title,
            section=section,
            text=text,
        ))
    return chunks


def split_documents(documents, max_chars=500):
    """批量切分文档。"""
    chunks = []
    for doc in documents:
        chunks.extend(split_document(doc, max_chars=max_chars))
    return chunks


def knowledge_signature(knowledge_dir):
    """计算知识库内容指纹（文件名 + 内容的 MD5），用于索引缓存失效判断。"""
    knowledge_dir = Path(knowledge_dir)
    md5 = hashlib.md5()
    for path in sorted(knowledge_dir.glob("*.md")):
        md5.update(path.name.encode("utf-8"))
        md5.update(path.read_bytes())
    return md5.hexdigest()
