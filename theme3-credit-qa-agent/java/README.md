# 主题3：基于 RAG 知识库的企业信贷问答 Agent（Java 版）

本案例以**真实 RAG（检索增强生成）组件链**实现产品知识库，替代硬编码字典与关键词模拟：

```text
knowledge/*.md 知识源文档
   → KBLoader（文档加载 + 章节级切分）
   → EmbeddingClient（OpenAI 兼容 Embedding API）
   → VectorIndex（内存余弦相似度索引）
   → Retriever（Top-K 语义检索 + 相似度阈值）
   → LLM 增强生成（无 Key 时降级为"检索结果组装"，未命中时转人工话术）
```

## 运行方式

### 直接运行（未配置 Embedding API 时自动降级为 BM25 稀疏检索）

```bash
cd java
javac -encoding UTF-8 *.java
java Main
```

### 启用 Embedding API（OpenAI 兼容）

```bash
# Linux / macOS
export EMBEDDING_API_KEY="你的密钥"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"   # 或行内嵌入服务地址
export EMBEDDING_MODEL="text-embedding-3-small"
javac -encoding UTF-8 *.java && java Main

# Windows PowerShell
$env:EMBEDDING_API_KEY="你的密钥"
javac -encoding UTF-8 *.java; java Main
```

### 启用 LLM 润色（可选，与 RAG 检索正交）

```bash
export LLM_API_KEY="你的API密钥"
export LLM_BASE_URL="https://api.moonshot.cn/v1"   # 默认，可选
export LLM_MODEL="moonshot-v1-8k"                  # 默认，可选
javac -encoding UTF-8 *.java && java Main
```

## 文件说明

| 文件 | 职责 |
|------|------|
| `Main.java` | 演示入口（含 RAG 检索轨迹展示） |
| `CreditQAAgent.java` | RAG Agent：上下文改写 → 检索 → 增强生成 → 引用溯源 |
| `RAGKnowledgeBase.java` | RAG 知识库装配层 |
| `KBLoader.java` | 文档加载 + 章节切分 |
| `EmbeddingClient.java` | 嵌入组件（OpenAI 兼容 API，标准库实现） |
| `VectorIndex.java` | 向量索引（内存余弦） |
| `BM25Index.java` | BM25 稀疏检索（降级/混合检索组件） |
| `Retriever.java` | 检索器（Top-K + 阈值 + 引用元数据） |
| `KBChunk.java` | 知识分片数据结构 |
| `Memory.java` | 对话记忆管理 |
| `Persistence.java` | 合规保存（知识库快照 + 对话记忆） |
| `LLMClient.java` | LLM 调用封装（标准库实现） |

知识库源文档位于案例根目录 `../knowledge/`（与 Python 版共用）。

## 环境要求

- JDK 8+，零第三方依赖，仅使用 Java 标准库
- 可选：Embedding API 环境变量、`LLM_API_KEY`

## 设计说明

- **知识库扩展零代码**：向 `knowledge/` 添加 Markdown 文档即可，重启后索引自动重建
- **降级链完整**：Embedding API → BM25 稀疏检索；LLM 润色 → 检索结果模板组装；未命中 → 转人工话术
- **引用溯源**：每条回答附信息来源（文档·章节），检索轨迹可见，满足银行审计要求
- **生产替换路径**：VectorIndex 可平替为 Milvus / pgvector / Elasticsearch（如结合 Spring AI 生态），上层接口不变
