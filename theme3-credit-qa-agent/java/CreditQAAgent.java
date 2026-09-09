import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Agent 核心逻辑模块：检索增强问答（RAG Agent）。
 *
 * 处理流程（标准 RAG 问答链路）：
 *
 *     用户问题
 *       → ① 上下文改写：结合最近对话历史构造检索查询（解决指代与省略）
 *       → ② 知识检索：Retriever 召回 Top-K 知识分片（含来源与相似度）
 *       → ③ 增强生成：检索结果 + 对话历史组装 Prompt，LLM 基于资料作答
 *                      （无 API Key 时降级为"检索结果直接组装"的模板回答）
 *       → ④ 引用溯源：回答末尾附信息来源，满足银行审计要求
 *       → ⑤ 未命中降级：相似度低于阈值时明确告知并建议转人工，不臆造答案
 */
public class CreditQAAgent {

    private static final String SYSTEM_PROMPT =
        "你是一位专业的银行信贷产品顾问。请严格基于给定的【检索资料】回答客户问题，" +
        "不得编造资料中没有的信息；若资料不足以回答，请直接说明并建议联系客户经理。" +
        "语气亲切专业，回答控制在 150 字以内；涉及材料清单时使用分点列表。";

    /** 意图 → 偏好章节：用于无 LLM 时从检索结果中选取最相关的章节内容。 */
    private static final String[][] SECTION_HINTS = {
        {"材料|资料|清单", "材料"},
        {"有哪些|哪些|列表", "产品体系|概述|简介"},
        {"额度|利率|期限|多少钱", "额度"},
        {"适合|对象|条件|准入", "适用对象"},
        {"流程|怎么办|如何申请", "流程"},
    };

    static final int RETRIEVE_TOP_K = 5; // 召回分片数（取宽，保证章节偏好选择有余地）
    static final int TRACE_TOP_K = 3;    // 检索轨迹与引用展示的分片数

    final RAGKnowledgeBase kb;
    final Memory memory;
    final LLMClient llm;
    private List<Retriever.RetrievalResult> lastRetrieval = new ArrayList<>();

    public CreditQAAgent() throws IOException {
        this.kb = new RAGKnowledgeBase();
        this.memory = new Memory();
        this.llm = new LLMClient();
    }

    // ---------- 对外接口 ----------

    public String ask(String question) {
        memory.addUserMessage(question);
        String retrievalQuery = buildRetrievalQuery(question);
        lastRetrieval = kb.retrieve(retrievalQuery, RETRIEVE_TOP_K);
        String answer = generate(question, lastRetrieval);
        memory.addAssistantMessage(answer);
        return answer;
    }

    public List<Retriever.RetrievalResult> getLastRetrieval() {
        return lastRetrieval;
    }

    public Memory getMemory() {
        return memory;
    }

    // ---------- RAG 链路 ----------

    /** ① 上下文改写：拼接最近的用户问题，消解"需要什么材料？"这类省略指代。 */
    private String buildRetrievalQuery(String question) {
        List<String> userHistory = new ArrayList<>();
        for (Map<String, String> m : memory.getMessages()) {
            if ("user".equals(m.get("role"))) {
                userHistory.add(m.get("content"));
            }
        }
        // 当前问题之前最近 2 轮用户提问
        int from = Math.max(0, userHistory.size() - 3);
        StringBuilder sb = new StringBuilder();
        for (int i = from; i < userHistory.size() - 1; i++) {
            sb.append(userHistory.get(i)).append(" ");
        }
        sb.append(question);
        return sb.toString().trim();
    }

    private String generate(String question, List<Retriever.RetrievalResult> results) {
        if (results.isEmpty()) {
            return noHitAnswer(); // ⑤ 未命中降级
        }

        // ③ 增强生成：LLM 优先，失败降级为模板组装
        String reply = llm.chat(SYSTEM_PROMPT, buildRagPrompt(question, results), 512);
        if (reply == null) {
            reply = templateAnswer(question, results);
        }

        // ④ 引用溯源
        StringBuilder sources = new StringBuilder();
        for (int i = 0; i < Math.min(TRACE_TOP_K, results.size()); i++) {
            String src = results.get(i).source();
            if (sources.indexOf(src) == -1) {
                if (sources.length() > 0) sources.append("、");
                sources.append(src);
            }
        }
        return reply + "\n（信息来源：" + sources + "）";
    }

    private String buildRagPrompt(String question, List<Retriever.RetrievalResult> results) {
        StringBuilder sb = new StringBuilder("【检索资料】\n");
        for (int i = 0; i < results.size(); i++) {
            Retriever.RetrievalResult r = results.get(i);
            sb.append("[资料").append(i + 1).append("] ").append(r.source())
              .append("（相似度 ").append(String.format("%.2f", r.score)).append("）\n")
              .append(r.chunk.text).append("\n\n");
        }
        sb.append("【对话历史】\n");
        List<Map<String, String>> msgs = memory.getMessages();
        for (int i = Math.max(0, msgs.size() - 4); i < msgs.size(); i++) {
            Map<String, String> m = msgs.get(i);
            sb.append("user".equals(m.get("role")) ? "客户：" : "顾问：")
              .append(m.get("content")).append("\n");
        }
        sb.append("\n【客户问题】\n").append(question).append("\n\n请仅依据检索资料回答。");
        return sb.toString();
    }

    /** 无 LLM 时的降级回答：直接基于检索命中的知识分片组装，答案有据可依。 */
    private String templateAnswer(String question, List<Retriever.RetrievalResult> results) {
        KBChunk chunk = pickSection(question, results);
        List<String> bullets = new ArrayList<>();
        for (String line : chunk.text.split("\n")) {
            String l = line.trim();
            if (l.startsWith("-")) {
                bullets.add(l.substring(1).trim());
            }
        }
        if (!bullets.isEmpty() && containsAny(question, "材料", "资料", "清单")) {
            StringBuilder sb = new StringBuilder("根据知识库检索结果，")
                    .append(chunk.docTitle).append("的相关要求如下：\n");
            for (int i = 0; i < bullets.size(); i++) {
                sb.append(i + 1).append(". ").append(bullets.get(i)).append("\n");
            }
            return sb.toString().trim();
        }
        return "根据知识库检索结果（" + chunk.docTitle + "·" + chunk.section + "）：" + chunk.text;
    }

    /** 按问题意图在检索结果中选取最相关章节（如"材料"问题优先选申请材料分片）。 */
    private static KBChunk pickSection(String question, List<Retriever.RetrievalResult> results) {
        for (String[] hint : SECTION_HINTS) {
            if (containsAny(question, hint[0].split("\\|"))) {
                for (Retriever.RetrievalResult r : results) {
                    if (containsAny(r.chunk.section, hint[1].split("\\|"))) {
                        return r.chunk;
                    }
                }
            }
        }
        return results.get(0).chunk;
    }

    /** ⑤ 未命中降级：不臆造，明确告知并给出转人工路径。 */
    private static String noHitAnswer() {
        return "抱歉，该问题暂未在当前知识库中检索到可靠答案，为避免误导，"
             + "建议您联系客户经理进一步确认。我们会将您的问题记录并补充进知识库。";
    }

    private static boolean containsAny(String text, String... keywords) {
        for (String k : keywords) {
            if (text.contains(k)) {
                return true;
            }
        }
        return false;
    }
}
