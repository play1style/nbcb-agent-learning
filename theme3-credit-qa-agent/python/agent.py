"""Agent 核心逻辑模块：检索增强问答（RAG Agent）。

处理流程（标准 RAG 问答链路）：

    用户问题
      → ① 上下文改写：结合最近对话历史构造检索查询（解决指代与省略）
      → ② 知识检索：Retriever 召回 Top-K 知识分片（含来源与相似度）
      → ③ 增强生成：检索结果 + 对话历史组装 Prompt，LLM 基于资料作答
                     （无 API Key 时降级为"检索结果直接组装"的模板回答）
      → ④ 引用溯源：回答末尾附信息来源，满足银行审计要求
      → ⑤ 未命中降级：相似度低于阈值时明确告知并建议转人工，不臆造答案
"""
from knowledge_base import RAGKnowledgeBase
from llm_client import LLMClient
from memory import Memory

# SYSTEM_PROMPT = (
#     "你是一位专业的银行信贷产品顾问。请严格基于给定的【检索资料】回答客户问题，"
#     "不得编造资料中没有的信息；若资料不足以回答，请直接说明并建议联系客户经理。"
#     "语气亲切专业，回答控制在 150 字以内；涉及材料清单时使用分点列表。"
# )

SYSTEM_PROMPT = (
    "你是一位专业的银行信贷产品顾问。请严格基于给定的【检索资料】回答客户问题，"
    "不得编造资料中没有的信息；若资料不足以回答，请直接说明并建议联系客户经理。"
    "语气亲切专业，回答控制在 150 字以内；**必须严格使用固定输出格式：**\n"
    "根据知识库检索结果，{产品名}的相关要求如下：\n"
    "1.材料一\n"
    "2.材料二\n"
    "... ...\n"
    "仅当资料无对应信息时，不使用该列表格式，直接说明信息不足，建议联系客户经理。"
)

# 意图 → 偏好章节：用于无 LLM 时从检索结果中选取最相关的章节内容
SECTION_HINTS = [
    (("材料", "资料", "清单"), ("材料",)),
    (("有哪些", "哪些", "列表"), ("产品体系", "概述", "简介")),
    (("额度", "利率", "期限", "多少钱"), ("额度",)),
    (("适合", "对象", "条件", "准入"), ("适用对象",)),
    (("流程", "怎么办", "如何申请"), ("流程",)),
]

RETRIEVE_TOP_K = 5       # 召回分片数（取宽，保证章节偏好选择有余地）
TRACE_TOP_K = 3          # 检索轨迹与引用展示的分片数


class CreditQAAgent:
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base or RAGKnowledgeBase()
        self.memory = Memory()
        self.llm = LLMClient()
        self.last_retrieval = []  # 最近一次检索轨迹，供演示与审计查看

    # ---------- 对外接口 ----------

    def ask(self, question):
        """处理用户问题并返回答复。"""
        self.memory.add_user_message(question)
        retrieval_query = self._build_retrieval_query(question)
        self.last_retrieval = self.kb.retrieve(retrieval_query, top_k=RETRIEVE_TOP_K)
        answer = self._generate(question, self.last_retrieval)
        self.memory.add_assistant_message(answer)
        return answer

    def get_memory(self):
        """返回所有对话记录，供持久化使用。"""
        return self.memory.get_messages()

    # ---------- RAG 链路 ----------

    def _build_retrieval_query(self, question):
        """上下文改写：拼接最近的用户问题，消解"需要什么材料？"这类省略指代。"""
        history = [m["content"] for m in self.memory.get_messages()
                   if m["role"] == "user"]
        context = history[-3:-1]  # 当前问题之前最近 2 轮用户提问
        return " ".join(context + [question]) if context else question

    def _generate(self, question, results):
        if not results:
            return self._no_hit_answer()

        # ③ 增强生成：LLM 优先，失败降级为模板组装
        reply = self.llm.chat(
            SYSTEM_PROMPT,
            self._build_rag_prompt(question, results),
            max_tokens=512,
        )
        if not reply:
            reply = self._template_answer(question, results)

        # ④ 引用溯源
        sources = "、".join(dict.fromkeys(r.source for r in results[:TRACE_TOP_K]))
        return f"{reply}\n（信息来源：{sources}）"

    def _build_rag_prompt(self, question, results):
        context_blocks = []
        for i, r in enumerate(results, 1):
            context_blocks.append(
                f"[资料{i}] {r.source}（相似度 {r.score:.2f}）\n{r.chunk.text}"
            )
        history = "\n".join(
            f"{'客户' if m['role'] == 'user' else '顾问'}：{m['content']}"
            for m in self.memory.get_messages()[-4:]
        )
        return (
            "【检索资料】\n" + "\n\n".join(context_blocks)
            + "\n\n【对话历史】\n" + history
            + "\n\n【客户问题】\n" + question
            + "\n\n请仅依据检索资料回答。"
        )

    def _template_answer(self, question, results):
        """无 LLM 时的降级回答：直接基于检索命中的知识分片组装，答案有据可依。"""
        chunk = self._pick_section(question, results)
        lines = [l.strip() for l in chunk.text.splitlines() if l.strip()]
        bullets = [l.lstrip("- ").strip() for l in lines if l.startswith("-")]

        if bullets and any(k in question for k in ("材料", "资料", "清单")):
            items = "\n".join(f"{i}. {b}" for i, b in enumerate(bullets, 1))
            body = f"根据知识库检索结果，{chunk.doc_title}的相关要求如下：\n{items}"
        else:
            body = (f"根据知识库检索结果（{chunk.doc_title}·{chunk.section}）："
                    f"{chunk.text}")
        return body

    @staticmethod
    def _pick_section(question, results):
        """按问题意图在检索结果中选取最相关章节（如"材料"问题优先选申请材料分片）。"""
        for keywords, sections in SECTION_HINTS:
            if any(k in question for k in keywords):
                for r in results:
                    if any(s in r.chunk.section for s in sections):
                        return r.chunk
        return results[0].chunk

    @staticmethod
    def _no_hit_answer():
        """⑤ 未命中降级：不臆造，明确告知并给出转人工路径。"""
        return (
            "抱歉，该问题暂未在当前知识库中检索到可靠答案，为避免误导，"
            "建议您联系客户经理进一步确认。我们会将您的问题记录并补充进知识库。"
        )
