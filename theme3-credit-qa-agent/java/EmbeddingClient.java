import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * RAG 组件：文本嵌入客户端（Embedding）。
 *
 * 通过 OpenAI 兼容接口（POST {baseUrl}/embeddings）将文本映射为稠密向量，
 * 适配行内统一部署的 Embedding 服务。仅使用 Java 标准库，零第三方依赖。
 *
 * 环境变量（均为可选）：
 *   EMBEDDING_API_KEY   – Embedding API 密钥（缺省回退读取 LLM_API_KEY）
 *   EMBEDDING_BASE_URL  – Embedding API 基地址（缺省回退读取 LLM_BASE_URL）
 *   EMBEDDING_MODEL     – 嵌入模型名称，默认 text-embedding-3-small
 *   EMBEDDING_TIMEOUT   – 请求超时毫秒数，默认 15000
 *
 * create() 在密钥缺失或服务不可用时返回 null，由上层降级为 BM25 稀疏检索。
 */
public class EmbeddingClient {

    private final String apiKey;
    private final String baseUrl;
    private final String model;
    private final int timeout;

    private EmbeddingClient(String apiKey, String baseUrl, String model, int timeout) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.model = model;
        this.timeout = timeout;
    }

    /** 探测并创建嵌入客户端；不可用时返回 null。 */
    public static EmbeddingClient create() {
        String apiKey = getenv2("EMBEDDING_API_KEY", "LLM_API_KEY").trim();
        if (apiKey.isEmpty()) {
            System.out.println("[RAG] 未配置 Embedding API 密钥，降级为 BM25 稀疏检索（生产环境常用混合检索组件）");
            return null;
        }
        String baseUrl = getenv2("EMBEDDING_BASE_URL", "LLM_BASE_URL").trim();
        if (baseUrl.isEmpty()) {
            baseUrl = "https://api.openai.com/v1";
        }
        baseUrl = baseUrl.replaceAll("/+$", "");
        String model = System.getenv().getOrDefault("EMBEDDING_MODEL", "text-embedding-3-small");
        int timeout = Integer.parseInt(System.getenv().getOrDefault("EMBEDDING_TIMEOUT", "15000"));

        EmbeddingClient client = new EmbeddingClient(apiKey, baseUrl, model, timeout);
        try {
            client.embed(java.util.Collections.singletonList("探测")); // 探测服务可用性
            System.out.println("[RAG] 嵌入组件：" + client.name());
            return client;
        } catch (Exception e) {
            System.out.println("[RAG] Embedding API 不可用（" + e.getMessage() + "），降级为 BM25 稀疏检索");
            return null;
        }
    }

    public String name() {
        return "embedding-api:" + model;
    }

    /** texts -> 归一化向量数组（L2 归一化后点积即余弦相似度）。 */
    public double[][] embed(List<String> texts) throws Exception {
        HttpURLConnection conn = (HttpURLConnection) URI.create(baseUrl + "/embeddings")
                .toURL().openConnection();
        try {
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Authorization", "Bearer " + apiKey);
            conn.setDoOutput(true);
            conn.setConnectTimeout(timeout);
            conn.setReadTimeout(timeout);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(buildJsonBody(texts).getBytes(StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    code >= 200 && code < 300 ? conn.getInputStream() : conn.getErrorStream(),
                    StandardCharsets.UTF_8));
            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            reader.close();
            if (code < 200 || code >= 300) {
                throw new RuntimeException("HTTP " + code + ": " + response);
            }
            return parseEmbeddings(response.toString(), texts.size());
        } finally {
            conn.disconnect();
        }
    }

    private String buildJsonBody(List<String> texts) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"model\":\"").append(escapeJson(model)).append("\",\"input\":[");
        for (int i = 0; i < texts.size(); i++) {
            if (i > 0) {
                sb.append(",");
            }
            sb.append("\"").append(escapeJson(texts.get(i))).append("\"");
        }
        sb.append("]}");
        return sb.toString();
    }

    /** 从响应中按顺序提取全部 "embedding":[...] 向量。 */
    private static double[][] parseEmbeddings(String json, int expected) {
        List<double[]> vectors = new ArrayList<>();
        String marker = "\"embedding\":[";
        int pos = 0;
        while (true) {
            int start = json.indexOf(marker, pos);
            if (start == -1) {
                break;
            }
            start += marker.length();
            int end = json.indexOf(']', start);
            String[] parts = json.substring(start, end).split(",");
            double[] vec = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                vec[i] = Double.parseDouble(parts[i].trim());
            }
            vectors.add(normalize(vec));
            pos = end + 1;
        }
        if (vectors.size() != expected) {
            throw new RuntimeException("Embedding 响应数量不符：期望 " + expected + "，实际 " + vectors.size());
        }
        return vectors.toArray(new double[0][]);
    }

    private static double[] normalize(double[] vec) {
        double norm = 0;
        for (double v : vec) {
            norm += v * v;
        }
        norm = Math.sqrt(norm);
        if (norm == 0) {
            return vec;
        }
        for (int i = 0; i < vec.length; i++) {
            vec[i] /= norm;
        }
        return vec;
    }

    private static String getenv2(String primary, String fallback) {
        String v = System.getenv(primary);
        if (v == null || v.trim().isEmpty()) {
            v = System.getenv().getOrDefault(fallback, "");
        }
        return v == null ? "" : v;
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
