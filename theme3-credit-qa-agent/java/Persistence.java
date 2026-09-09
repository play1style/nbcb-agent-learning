import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/**
 * 配置持久化模块。
 *
 * 演示如何保存 Agent 的知识库快照（RAG 配置）与对话记忆，
 * 满足银行合规保存要求：可备份、可审计、可迁移。
 */
public class Persistence {

    public void save(String filepath, Map<String, String> kbStats,
                     List<String> docIds, List<Map<String, String>> messages) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Agent 配置备份 ===\n\n");

        sb.append("[知识库快照（RAG）]\n");
        sb.append("知识库目录：").append(kbStats.get("knowledgeDir")).append("\n");
        sb.append("文档数：").append(kbStats.get("documentCount"))
          .append("，分片数：").append(kbStats.get("chunkCount")).append("\n");
        for (String id : docIds) {
            sb.append("- ").append(id).append("\n");
        }
        sb.append("嵌入组件：").append(kbStats.get("embedding")).append("\n");
        sb.append("检索模式：").append(kbStats.get("retrievalMode"))
          .append("，相似度阈值：").append(kbStats.get("minScore")).append("\n\n");

        sb.append("[对话记忆]\n");
        for (Map<String, String> m : messages) {
            sb.append(m.get("role")).append(": ").append(m.get("content")).append("\n");
        }

        Files.write(Paths.get(filepath), sb.toString().getBytes(StandardCharsets.UTF_8));
        System.out.println("配置已保存到：" + filepath);
    }

    public void load(String filepath) {
        System.out.println("已从 " + filepath + " 恢复配置（演示用）");
    }
}
