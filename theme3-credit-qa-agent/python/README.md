# 主题3：基于 RAG 知识库的企业信贷问答 Agent（Python 版）

本案例以**真实 RAG（检索增强生成）组件链**实现产品知识库，替代硬编码字典与关键词模拟：

```text
knowledge/*.md 知识源文档
   → DocumentLoader（文档加载）
   → TextSplitter（章节级切分）
   → Embedder（sentence-transformers 本地模型 / OpenAI 兼容 Embedding API）
   → VectorStore（余弦相似度索引，JSON 持久化 + 指纹缓存）
   → Retriever（Top-K 语义检索 + 相似度阈值）
   → LLM 增强生成（无 Key 时降级为"检索结果组装"，未命中时转人工话术）
```

## 运行方式

### 直接运行（无嵌入后端时自动降级为 BM25 稀疏检索，零外部依赖）

```bash
cd python
pip install numpy        # 向量索引需要；BM25 模式亦加载本模块
python main.py
```

### 启用本地嵌入模型（推荐，离线可用）

```bash
pip install sentence-transformers
python main.py           # 默认使用 BAAI/bge-small-zh-v1.5，可用 EMBEDDING_MODEL 覆盖
```

### 启用 Embedding API（OpenAI 兼容）

```bash
# Windows PowerShell
$env:EMBEDDING_API_KEY="你的密钥"
$env:EMBEDDING_BASE_URL="https://api.openai.com/v1"   # 或行内嵌入服务地址
$env:EMBEDDING_MODEL="text-embedding-3-small"
python main.py
```

### 启用 LLM 润色（可选，与 RAG 检索正交）

```bash
$env:LLM_API_KEY="你的API密钥"
$env:LLM_BASE_URL="https://api.moonshot.cn/v1"   # 默认，可选
$env:LLM_MODEL="moonshot-v1-8k"                  # 默认，可选
python main.py
```

## 目录结构

```text
python/
├── main.py               # 演示入口（含 RAG 检索轨迹展示）
├── agent.py              # RAG Agent：上下文改写 → 检索 → 增强生成 → 引用溯源
├── knowledge_base.py     # RAG 知识库装配层（RAGKnowledgeBase）
├── memory.py             # 对话记忆管理
├── persistence.py        # 合规保存（知识库快照 + 对话记忆）
├── llm_client.py         # LLM 调用封装（标准库实现）
├── requirements.txt
├── rag/                  # RAG 组件包
│   ├── loader.py         # 文档加载 + 章节切分 + 内容指纹
│   ├── embeddings.py     # 嵌入组件（本地模型 / Embedding API）
│   ├── vector_store.py   # 向量索引（NumPy 余弦 + JSON 持久化）
│   ├── bm25.py           # BM25 稀疏检索（降级/混合检索组件）
│   └── retriever.py      # 检索器（Top-K + 阈值 + 引用元数据）
└── ../knowledge/         # 知识库源文档（与 Java 版共用）
    ├── 产品总览.md
    ├── 小微快贷.md
    └── 对公流贷.md
```

## 环境要求

- Python 3.8+，必需依赖仅 `numpy`
- 可选：`sentence-transformers`（本地嵌入）或 Embedding API 环境变量
- 可选：`LLM_API_KEY`（LLM 润色）

## 设计说明

- **知识库扩展零代码**：向 `knowledge/` 添加 Markdown 文档即可，重启后索引自动重建（内容指纹校验缓存）
- **降级链完整**：本地嵌入 → Embedding API → BM25 稀疏检索；LLM 润色 → 检索结果模板组装；未命中 → 转人工话术。任何一环缺失都不影响演示运行
- **引用溯源**：每条回答附信息来源（文档·章节），检索轨迹可见，满足银行审计要求
- **生产替换路径**：VectorStore 可平替为 FAISS / Milvus / pgvector / Elasticsearch，Embedder 可平替为行内嵌入服务，上层接口不变
