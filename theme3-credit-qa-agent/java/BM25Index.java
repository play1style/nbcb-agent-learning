import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * RAG 组件：BM25 稀疏检索索引（无嵌入后端时的生产级降级方案）。
 *
 * BM25（Okapi BM25）是 Elasticsearch、Lucene 等生产搜索引擎的默认排序算法，
 * 在真实 RAG 系统中常与向量检索组成"混合检索（Hybrid Search）"。
 * 本实现仅依赖 Java 标准库，采用中文字符二元组（bigram）分词。
 */
public class BM25Index {

    private final double k1;
    private final double b;
    private List<KBChunk> chunks = new ArrayList<>();
    private final List<Map<String, Integer>> docTf = new ArrayList<>();
    private final List<Integer> docLen = new ArrayList<>();
    private final Map<String, Integer> df = new HashMap<>();
    private double avgLen = 0;

    public BM25Index() {
        this(1.5, 0.75); // 业界常用参数
    }

    public BM25Index(double k1, double b) {
        this.k1 = k1;
        this.b = b;
    }

    /** 对分片列表建立索引；标题与章节词权重高于正文（重复计入词频实现加权）。 */
    public BM25Index build(List<KBChunk> chunkList) {
        this.chunks = new ArrayList<>(chunkList);
        for (KBChunk c : chunks) {
            StringBuilder weighted = new StringBuilder();
            for (int i = 0; i < 3; i++) weighted.append(c.docTitle).append(' ');
            for (int i = 0; i < 2; i++) weighted.append(c.section).append(' ');
            weighted.append(c.text);
            Map<String, Integer> tf = termFreq(tokenize(weighted.toString()));
            docTf.add(tf);
            int len = 0;
            for (int v : tf.values()) len += v;
            docLen.add(len);
            for (String term : tf.keySet()) {
                df.merge(term, 1, Integer::sum);
            }
        }
        int total = 0;
        for (int l : docLen) total += l;
        avgLen = chunks.isEmpty() ? 0 : (double) total / chunks.size();
        return this;
    }

    /** 返回按 BM25 得分降序的 (chunk, score) 列表，仅保留正得分。 */
    public List<Retriever.RetrievalResult> search(String query, int topK) {
        double[] total = new double[chunks.size()];
        for (String term : new java.util.HashSet<>(tokenize(query))) {
            Integer dfVal = df.get(term);
            if (dfVal == null || dfVal == 0) continue;
            double idf = Math.log(1 + (chunks.size() - dfVal + 0.5) / (dfVal + 0.5));
            for (int i = 0; i < chunks.size(); i++) {
                Integer f = docTf.get(i).get(term);
                if (f == null || f == 0) continue;
                double norm = 1 - b + b * docLen.get(i) / (avgLen == 0 ? 1 : avgLen);
                total[i] += idf * f * (k1 + 1) / (f + k1 * norm);
            }
        }
        List<Retriever.RetrievalResult> results = new ArrayList<>();
        for (int i = 0; i < chunks.size(); i++) {
            if (total[i] > 0) {
                results.add(new Retriever.RetrievalResult(chunks.get(i), total[i]));
            }
        }
        results.sort((a, b2) -> Double.compare(b2.score, a.score));
        return results.subList(0, Math.min(topK, results.size()));
    }

    /** 中文字符二元组 + 英文数字单词的混合分词（与 Python 版一致）。 */
    static List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        String lower = text.toLowerCase();
        StringBuilder word = new StringBuilder();
        List<Character> cjk = new ArrayList<>();
        for (int i = 0; i < lower.length(); i++) {
            char c = lower.charAt(i);
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
                word.append(c);
            } else {
                if (word.length() > 0) {
                    tokens.add(word.toString());
                    word.setLength(0);
                }
                if (c >= 0x4E00 && c <= 0x9FFF) {
                    cjk.add(c);
                }
            }
        }
        if (word.length() > 0) {
            tokens.add(word.toString());
        }
        for (int i = 0; i + 1 < cjk.size(); i++) {
            tokens.add("" + cjk.get(i) + cjk.get(i + 1));
        }
        for (char c : cjk) {
            tokens.add(String.valueOf(c)); // 单字兜底，提升短查询召回
        }
        return tokens;
    }

    private static Map<String, Integer> termFreq(List<String> tokens) {
        Map<String, Integer> tf = new HashMap<>();
        for (String t : tokens) {
            tf.merge(t, 1, Integer::sum);
        }
        return tf;
    }
}
