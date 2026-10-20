"""重建 ``vector_store.json`` 的命令行工具。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 直接执行 scripts/build_index.py 时，将项目根目录加入模块搜索路径。
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings
from app.embeddings import create_embedder
from app.knowledge_base import VectorStore


def main() -> None:
    """从全部 Markdown 制度文件构建向量索引。"""
    settings = Settings.from_env()
    settings.ensure_directories()
    embedder = create_embedder(settings)
    store = VectorStore(settings, embedder)
    store.build(force=True)
    print(
        "索引构建完成："
        f"documents={store.metadata['document_count']}, "
        f"chunks={store.metadata['chunk_count']}, "
        f"backend={store.metadata['embedding_backend']}, "
        f"path={store.path}"
    )


if __name__ == "__main__":
    main()
