/**
 * RAG 组件：知识分片数据结构。
 *
 * Chunk 是文档切分后的最小检索单元，携带来源元数据，用于检索命中后的引用溯源。
 */
public class KBChunk {
    public final String chunkId;   // 分片唯一标识（docId#序号）
    public final String docId;     // 所属文档
    public final String docTitle;  // 所属文档标题（用于引用展示）
    public final String section;   // 所属章节（如"申请材料"）
    public final String text;      // 分片正文

    public KBChunk(String chunkId, String docId, String docTitle, String section, String text) {
        this.chunkId = chunkId;
        this.docId = docId;
        this.docTitle = docTitle;
        this.section = section;
        this.text = text;
    }
}
