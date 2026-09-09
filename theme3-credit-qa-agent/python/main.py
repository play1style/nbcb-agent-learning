"""主题3：基于 RAG 知识库的带记忆企业信贷问答 Agent（Python 版）

演示入口：装配 RAG 知识库 → 三轮多轮对话（含检索轨迹展示）→ 合规保存。
"""
from agent import CreditQAAgent
from persistence import Persistence


def main():
    agent = CreditQAAgent()
    persistence = Persistence()

    questions = [
        "宁波银行有哪些企业贷款？",
        "小微企业适合哪个？",
        "需要什么材料？",
    ]

    for q in questions:
        print(f"用户：{q}")
        answer = agent.ask(q)
        print(_format_trace(agent.last_retrieval))
        print(f"Agent：\n{answer}\n")

    persistence.save("agent_config.txt", agent.kb.stats(), agent.get_memory())


def _format_trace(results, show=3):
    """格式化检索轨迹，直观展示 RAG 的检索过程（展示前 show 条）。"""
    if not results:
        return "  [RAG 检索] 未命中任何知识分片（低于相似度阈值）"
    items = "；".join(f"《{r.chunk.doc_title}》·{r.chunk.section} ({r.score:.2f})"
                      for r in results[:show])
    return f"  [RAG 检索] 命中 {len(results)} 条知识分片，Top{show}：{items}"


if __name__ == "__main__":
    main()
