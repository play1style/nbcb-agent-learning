"""配置持久化模块。

演示如何保存 Agent 的知识库快照（RAG 配置）与对话记忆，
满足银行合规保存要求：可备份、可审计、可迁移。
"""


class Persistence:
    def save(self, filepath, kb_stats, messages):
        """保存知识库快照与对话记忆到文件。"""
        lines = ["=== Agent 配置备份 ===", ""]

        lines.append("[知识库快照（RAG）]")
        lines.append(f"知识库目录：{kb_stats['knowledge_dir']}")
        lines.append(f"文档数：{kb_stats['document_count']}，分片数：{kb_stats['chunk_count']}")
        for doc_id in kb_stats["documents"]:
            lines.append(f"- {doc_id}")
        lines.append(f"嵌入组件：{kb_stats['embedding']}")
        lines.append(f"检索模式：{kb_stats['retrieval_mode']}，相似度阈值：{kb_stats['min_score']}")
        if kb_stats.get("index_path"):
            lines.append(f"向量索引文件：{kb_stats['index_path']}")
        lines.append("")

        lines.append("[对话记忆]")
        for m in messages:
            lines.append(f"{m['role']}: {m['content']}")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"配置已保存到：{filepath}")

    def load(self, filepath):
        """从文件恢复配置（演示用）。"""
        print(f"已从 {filepath} 恢复配置（演示用）")
