import java.util.List;

/**
 * 主题3：基于 RAG 知识库的带记忆企业信贷问答 Agent（Java 版）
 *
 * 演示入口：装配 RAG 知识库 → 三轮多轮对话（含检索轨迹展示）→ 合规保存。
 */
public class Main {
    public static void main(String[] args) throws Exception {
        CreditQAAgent agent = new CreditQAAgent();
        Persistence persistence = new Persistence();

        String[] questions = {
            "宁波银行有哪些企业贷款？",
            "小微企业适合哪个？",
            "需要什么材料？",
        };

        for (String q : questions) {
            System.out.println("用户：" + q);
            String answer = agent.ask(q);
            System.out.println(formatTrace(agent.getLastRetrieval(), CreditQAAgent.TRACE_TOP_K));
            System.out.println("Agent：\n" + answer + "\n");
        }

        persistence.save("agent_config.txt", agent.kb.stats(),
                agent.kb.docIds(), agent.getMemory().getMessages());
    }

    /** 格式化检索轨迹，直观展示 RAG 的检索过程（展示前 show 条）。 */
    static String formatTrace(List<Retriever.RetrievalResult> results, int show) {
        if (results.isEmpty()) {
            return "  [RAG 检索] 未命中任何知识分片（低于相似度阈值）";
        }
        StringBuilder sb = new StringBuilder();
        sb.append("  [RAG 检索] 命中 ").append(results.size())
          .append(" 条知识分片，Top").append(show).append("：");
        for (int i = 0; i < Math.min(show, results.size()); i++) {
            Retriever.RetrievalResult r = results.get(i);
            if (i > 0) sb.append("；");
            sb.append(r.source()).append(" (").append(String.format("%.2f", r.score)).append(")");
        }
        return sb.toString();
    }
}
