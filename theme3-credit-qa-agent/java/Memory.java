import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 对话记忆模块。
 *
 * 保存用户问题和 Agent 回答，为上下文理解提供支持。
 */
public class Memory {
    List<Map<String, String>> messages = new ArrayList<>();

    public void addUserMessage(String content) {
        messages.add(makeMessage("user", content));
    }

    public void addAssistantMessage(String content) {
        messages.add(makeMessage("assistant", content));
    }

    public String getUserHistory() {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            if ("user".equals(m.get("role"))) {
                sb.append(m.get("content"));
            }
        }
        return sb.toString();
    }

    public List<Map<String, String>> getMessages() {
        return messages;
    }

    private Map<String, String> makeMessage(String role, String content) {
        Map<String, String> m = new HashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }
}
