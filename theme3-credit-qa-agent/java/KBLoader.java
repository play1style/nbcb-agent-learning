import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * RAG 组件：文档加载与切分（Document Loader + Text Splitter）。
 *
 * - 从知识库目录加载全部 Markdown 源文档
 * - 按二级标题（## 章节）切分为检索分片；超长章节按段落二次切分
 */
public class KBLoader {

    private static final int MAX_CHARS = 500;
    private static final Pattern H1 = Pattern.compile("(?m)^#\\s+(.+)$");
    private static final Pattern H2_SPLIT = Pattern.compile("(?m)^##\\s+");
    private static final Pattern PARA_SPLIT = Pattern.compile("\\n\\s*\\n");

    /** 加载目录下全部 .md 文档并切分为分片列表。 */
    public static List<KBChunk> loadChunks(Path knowledgeDir) throws IOException {
        if (!Files.isDirectory(knowledgeDir)) {
            throw new IOException("知识库目录不存在：" + knowledgeDir);
        }
        // 按文件名排序，保证切分结果稳定
        TreeSet<String> names = new TreeSet<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(knowledgeDir, "*.md")) {
            for (Path p : stream) {
                names.add(p.getFileName().toString());
            }
        }
        if (names.isEmpty()) {
            throw new IOException("知识库目录 " + knowledgeDir + " 下没有可用的 .md 文档");
        }

        List<KBChunk> chunks = new ArrayList<>();
        List<String> docIds = new ArrayList<>();
        for (String name : names) {
            Path path = knowledgeDir.resolve(name);
            String content = new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
            String docId = name.replaceFirst("\\.md$", "");
            docIds.add(docId);
            Matcher m = H1.matcher(content);
            String title = m.find() ? m.group(1).trim() : docId;
            chunks.addAll(splitDocument(docId, title, content));
        }
        return chunks;
    }

    /** 将一份文档切分为分片（与 Python 版 loader.py 逻辑一致）。 */
    static List<KBChunk> splitDocument(String docId, String title, String content) {
        List<KBChunk> chunks = new ArrayList<>();
        String[] sections = H2_SPLIT.split(content);

        List<String[]> pending = new ArrayList<>(); // [section, text]
        for (int i = 0; i < sections.length; i++) {
            String sec = sections[i].trim();
            if (sec.isEmpty()) {
                continue;
            }
            if (i == 0) {
                // 文档开头（含 # 标题），去掉标题行后作为"概述"分片
                String text = content.split("\n").length > 0
                        ? sec.replaceAll("(?m)^#\\s+.+$", "").trim() : "";
                if (!text.isEmpty()) {
                    pending.add(new String[]{"概述", text});
                }
                continue;
            }
            int nl = sec.indexOf('\n');
            String sectionName = nl > 0 ? sec.substring(0, nl).trim() : sec;
            String body = nl > 0 ? sec.substring(nl + 1).trim() : "";
            if (body.isEmpty()) {
                continue;
            }
            if (body.length() <= MAX_CHARS) {
                pending.add(new String[]{sectionName, body});
            } else {
                String[] paras = PARA_SPLIT.split(body);
                for (int j = 0, k = 0; j < paras.length; j++) {
                    String para = paras[j].trim();
                    if (!para.isEmpty()) {
                        String suffix = k > 0 ? "（" + (k + 1) + "）" : "";
                        pending.add(new String[]{sectionName + suffix, para});
                        k++;
                    }
                }
            }
        }

        for (int idx = 0; idx < pending.size(); idx++) {
            String[] p = pending.get(idx);
            chunks.add(new KBChunk(docId + "#" + idx, docId, title, p[0], p[1]));
        }
        return chunks;
    }
}
