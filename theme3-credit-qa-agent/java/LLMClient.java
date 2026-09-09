import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;

/**
 * LLM 客户端（主题2 & 主题3 通用）。
 *
 * 使用 Java 标准库 HttpURLConnection 调用 OpenAI 兼容格式的 LLM API，
 * 支持真实调用与 Mock 自动降级两种模式。
 *
 * 环境变量（均为可选）：
 *   LLM_API_KEY    – API 密钥，未设置时自动进入 Mock 模式
 *   LLM_BASE_URL   – API 基地址，默认 https://api.moonshot.cn/v1
 *   LLM_MODEL      – 模型名称，默认 moonshot-v1-8k
 *   LLM_TIMEOUT    – 请求超时毫秒数，默认 10000
 *   LLM_MOCK       – mock 控制：auto（默认）| true | false
 *
 * 使用示例：
 *     LLMClient client = new LLMClient();
 *     String reply = client.chat("你是银行信贷专家", "请分析以下企业...");
 *     if (reply == null) {
 *         // 降级到模板/规则逻辑
 *     }
 */
public class LLMClient {

    private final String apiKey;
    private final String baseUrl;
    private final String model;
    private final int timeout;
    private final String mockMode;

    public LLMClient() {
        this.apiKey = System.getenv().getOrDefault("LLM_API_KEY", "").trim();
        this.baseUrl = System.getenv().getOrDefault("LLM_BASE_URL", "https://api.moonshot.cn/v1").replaceAll("/+$", "");
        this.model = System.getenv().getOrDefault("LLM_MODEL", "moonshot-v1-8k");
        this.timeout = Integer.parseInt(System.getenv().getOrDefault("LLM_TIMEOUT", "10000"));
        this.mockMode = System.getenv().getOrDefault("LLM_MOCK", "auto").toLowerCase();
    }

    public boolean isMock() {
        if ("true".equals(mockMode)) {
            return true;
        }
        if ("false".equals(mockMode)) {
            return false;
        }
        // auto：没有 API Key 时自动 mock
        return apiKey.isEmpty();
    }

    /**
     * 调用 LLM 生成文本。
     *
     * @param systemPrompt 系统提示词
     * @param userPrompt   用户提示词
     * @param maxTokens    最大生成 token 数
     * @return 生成文本；若处于 Mock 模式或调用失败则返回 null
     */
    public String chat(String systemPrompt, String userPrompt, int maxTokens) {
        if (isMock()) {
            return null;
        }

        HttpURLConnection conn = null;
        try {
            conn = (HttpURLConnection) URI.create(baseUrl + "/chat/completions")
                    .toURL().openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Authorization", "Bearer " + apiKey);
            conn.setDoOutput(true);
            conn.setConnectTimeout(timeout);
            conn.setReadTimeout(timeout);

            String jsonBody = buildJsonBody(systemPrompt, userPrompt, maxTokens);
            try (OutputStream os = conn.getOutputStream()) {
                os.write(jsonBody.getBytes(StandardCharsets.UTF_8));
            }

            int responseCode = conn.getResponseCode();
            BufferedReader reader;
            if (responseCode >= 200 && responseCode < 300) {
                reader = new BufferedReader(new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8));
            } else {
                reader = new BufferedReader(new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8));
                StringBuilder err = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    err.append(line);
                }
                System.out.println("[LLM 调用失败，自动降级到模板模式] HTTP " + responseCode + ": " + err);
                return null;
            }

            StringBuilder response = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                response.append(line);
            }
            reader.close();

            return extractContent(response.toString());

        } catch (Exception e) {
            System.out.println("[LLM 调用失败，自动降级到模板模式] 错误: " + e.getMessage());
            return null;
        } finally {
            if (conn != null) {
                conn.disconnect();
            }
        }
    }

    public String chat(String systemPrompt, String userPrompt) {
        return chat(systemPrompt, userPrompt, 512);
    }

    private String buildJsonBody(String systemPrompt, String userPrompt, int maxTokens) {
        String sys = escapeJson(systemPrompt);
        String user = escapeJson(userPrompt);
        return "{"
            + "\"model\":\"" + model + "\","
            + "\"messages\":["
            + "{\"role\":\"system\",\"content\":\"" + sys + "\"},"
            + "{\"role\":\"user\",\"content\":\"" + user + "\"}"
            + "],"
            + "\"max_tokens\":" + maxTokens + ","
            + "\"temperature\":0.3"
            + "}";
    }

    private String extractContent(String json) {
        // 简单解析：找到 "content":"..." 并提取值
        String marker = "\"content\":\"";
        int start = json.indexOf(marker);
        if (start == -1) return null;
        start += marker.length();
        StringBuilder sb = new StringBuilder();
        for (int i = start; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '"' && (i == start || json.charAt(i - 1) != '\\')) {
                break;
            }
            if (c == '\\' && i + 1 < json.length()) {
                char next = json.charAt(i + 1);
                switch (next) {
                    case '"': sb.append('"'); i++; break;
                    case 'n': sb.append('\n'); i++; break;
                    case 't': sb.append('\t'); i++; break;
                    case 'r': sb.append('\r'); i++; break;
                    case '\\': sb.append('\\'); i++; break;
                    default: sb.append(next); i++; break;
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString().trim();
    }

    private String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
