import java.util.Iterator;
import java.util.List;

/**
 * RAG 组件：检索器（Retriever）。
 *
 * 对上层（Agent）提供统一的语义检索接口：
 * - 稠密检索：查询嵌入 + 向量索引近邻搜索 + 相似度阈值过滤
 * - 稀疏检索：无嵌入后端（或运行时嵌入调用失败）时降级为 BM25
 * - 每次检索返回带来源与得分的 RetrievalResult，支撑引用溯源与审计
 */
public class Retriever {

    /** 检索命中结果：知识分片 + 得分。 */
    public static class RetrievalResult {
        public final KBChunk chunk;
        public final double score;

        public RetrievalResult(KBChunk chunk, double score) {
            this.chunk = chunk;
            this.score = score;
        }

        /** 引用来源描述，如《小微快贷》·申请材料。 */
        public String source() {
            return "《" + chunk.docTitle + "》·" + chunk.section;
        }
    }

    private final List<KBChunk> chunks;
    private final EmbeddingClient embedder;
    private final VectorIndex vectorIndex;
    private final BM25Index bm25;
    private final double minScore;

    public Retriever(List<KBChunk> chunks, EmbeddingClient embedder,
                     VectorIndex vectorIndex, double minScore) {
        this.chunks = chunks;
        this.embedder = embedder;
        this.vectorIndex = vectorIndex;
        this.minScore = minScore;
        // 无嵌入后端时预建 BM25 索引；有嵌入后端时保留分片用于运行时降级
        this.bm25 = embedder == null ? new BM25Index().build(chunks) : null;
    }

    public String mode() {
        return embedder != null ? "dense" : "sparse(BM25)";
    }

    /** 检索与 query 最相关的知识分片。 */
    public List<RetrievalResult> retrieve(String query, int topK) {
        if (embedder != null) {
            try {
                double[][] qv = embedder.embed(java.util.Collections.singletonList(query));
                List<RetrievalResult> hits = vectorIndex.search(qv[0], topK);
                Iterator<RetrievalResult> it = hits.iterator();
                while (it.hasNext()) {
                    if (it.next().score < minScore) {
                        it.remove();
                    }
                }
                return hits;
            } catch (Exception e) {
                // 运行时嵌入失败：本轮降级为 BM25，保证服务不中断
                System.out.println("[RAG] 查询嵌入失败（" + e.getMessage() + "），本轮降级为 BM25");
                return new BM25Index().build(chunks).search(query, topK);
            }
        }
        return bm25.search(query, topK);
    }
}
