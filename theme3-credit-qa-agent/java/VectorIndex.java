import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * RAG 组件：向量索引（Vector Store，内存版）。
 *
 * 向量已做 L2 归一化，点积即余弦相似度；按得分降序返回 Top-K。
 * 本实现聚焦教学演示；生产环境可替换为 FAISS / Milvus / pgvector / Elasticsearch，
 * 上层 Retriever 接口保持不变。
 */
public class VectorIndex {

    private final List<KBChunk> chunks = new ArrayList<>();
    private final List<double[]> vectors = new ArrayList<>();

    public void add(KBChunk chunk, double[] vector) {
        chunks.add(chunk);
        vectors.add(vector);
    }

    public int size() {
        return chunks.size();
    }

    /** 余弦相似度近邻检索，返回按得分降序的 (chunk, score) 列表。 */
    public List<Retriever.RetrievalResult> search(double[] queryVector, int topK) {
        List<Retriever.RetrievalResult> all = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            double score = dot(vectors.get(i), queryVector);
            all.add(new Retriever.RetrievalResult(chunks.get(i), score));
        }
        all.sort((a, b) -> Double.compare(b.score, a.score));
        return all.subList(0, Math.min(topK, all.size()));
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }
}
