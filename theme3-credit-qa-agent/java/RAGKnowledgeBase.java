import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * RAG 知识库模块（知识中枢）。
 *
 * 将 RAG 各组件装配为一条完整的检索增强管线：
 *
 *     知识库目录（knowledge/*.md）
 *         → KBLoader         文档加载与章节级切分
 *         → EmbeddingClient  文本嵌入（OpenAI 兼容 Embedding API）
 *         → VectorIndex      向量索引构建（内存余弦索引）
 *         → Retriever        Top-K 语义检索（无嵌入后端时降级 BM25 稀疏检索）
 *
 * 扩展知识库只需向 knowledge/ 目录添加 Markdown 文档并重启，
 * 无需修改任何检索与问答代码——这正是 RAG 架构相对硬编码知识库的核心优势。
 */
public class RAGKnowledgeBase {

    private final Path knowledgeDir;
    private final List<KBChunk> chunks;
    private final EmbeddingClient embedder;
    private final Retriever retriever;
    private final double minScore;

    public RAGKnowledgeBase() throws IOException {
        this(resolveKnowledgeDir(),
             Double.parseDouble(System.getenv().getOrDefault("RAG_MIN_SCORE", "0.35")));
    }

    public RAGKnowledgeBase(Path knowledgeDir, double minScore) throws IOException {
        this.knowledgeDir = knowledgeDir;
        this.minScore = minScore;

        // 1. 文档加载与切分
        this.chunks = KBLoader.loadChunks(knowledgeDir);

        // 2. 嵌入组件（不可用时为 null，降级 BM25）
        this.embedder = EmbeddingClient.create();

        // 3. 向量索引：嵌入全部知识分片
        VectorIndex vectorIndex = null;
        if (embedder != null) {
            try {
                List<String> texts = new ArrayList<>();
                for (KBChunk c : chunks) {
                    // 嵌入文本 = 标题 + 章节 + 正文，提升检索相关性
                    texts.add(c.docTitle + "｜" + c.section + "｜" + c.text);
                }
                double[][] vectors = embedder.embed(texts);
                vectorIndex = new VectorIndex();
                for (int i = 0; i < chunks.size(); i++) {
                    vectorIndex.add(chunks.get(i), vectors[i]);
                }
                System.out.println("[RAG] 向量索引已构建（" + vectorIndex.size() + " 个分片）");
            } catch (Exception e) {
                System.out.println("[RAG] 索引构建失败（" + e.getMessage() + "），降级为 BM25 稀疏检索");
            }
        }

        // 4. 检索器
        this.retriever = new Retriever(chunks, vectorIndex != null ? embedder : null,
                                       vectorIndex, minScore);
        System.out.println("[RAG] 知识库就绪：" + docIds().size() + " 份文档 / "
                + chunks.size() + " 个分片 / 检索模式 " + retriever.mode());
    }

    /** 知识库目录：默认取案例根目录下的 knowledge/（与 Python 版共用）。 */
    private static Path resolveKnowledgeDir() {
        Path parent = Paths.get("..", "knowledge");
        if (java.nio.file.Files.isDirectory(parent)) {
            return parent;
        }
        return Paths.get("knowledge");
    }

    /** 语义检索与 query 最相关的知识分片（RetrievalResult 列表）。 */
    public List<Retriever.RetrievalResult> retrieve(String query, int topK) {
        return retriever.retrieve(query, topK);
    }

    /** 知识库运行状态，用于持久化备份与审计。 */
    public Map<String, String> stats() {
        Map<String, String> stats = new LinkedHashMap<>();
        stats.put("knowledgeDir", knowledgeDir.toString());
        stats.put("documentCount", String.valueOf(docIds().size()));
        stats.put("chunkCount", String.valueOf(chunks.size()));
        stats.put("embedding", embedder != null ? embedder.name() : "无（BM25 降级）");
        stats.put("retrievalMode", retriever.mode());
        stats.put("minScore", String.valueOf(minScore));
        return stats;
    }

    /** 知识库文档 ID 列表。 */
    public List<String> docIds() {
        List<String> ids = new ArrayList<>();
        for (KBChunk c : chunks) {
            if (!ids.contains(c.docId)) {
                ids.add(c.docId);
            }
        }
        return ids;
    }
}
