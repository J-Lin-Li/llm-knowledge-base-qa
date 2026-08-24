import uuid
from pprint import pprint

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from tracing.context import TraceContext
from tracing.handler import TraceHandler
from tracing.db import create_session, list_sessions
from utils.env_utils import POSTGRES_URI

# from draw_png import draw_graph
from graph2.assemble_context_node import assemble_context
from graph2.check_answer_node import check_answer
from graph2.check_hallucination_node import check_hallucination
from graph2.generate_node2 import generate
from graph2.grade_documents_node import grade_documents
from graph2.grade_web_results_node import grade_web_results
from graph2.graph_state2 import GraphState
from graph2.retriever_node import retrieve
from graph2.transform_query_node import transform_query
from graph2.web_search_node import web_search
from utils.log_utils import log


def route_from_hallucination(state):
    """
    幻觉检测后的路由：读取 check_hallucination 写入的结果和 generation_count。

    generation_count 已由 generate 节点自增，此处只做检查：
      - "yes"（无幻觉）→ check_answer
      - "no"（有幻觉）且 generation_count < 3 → generate 重试
      - "no"（有幻觉）且 generation_count >= 3 → hallucination_fallback（兜底）
    """
    generation_count = state.get("generation_count", 0)
    hallucination_result = state.get("hallucination_result", "no")

    if hallucination_result == "yes":
        log.info("---判定：生成内容基于参考文档，进入答案质量评估---")
        return "check_answer"
    elif generation_count >= 3:
        log.info("---幻觉检测已重试3次，强制结束---")
        return "hallucination_fallback"
    else:
        log.info(f"---判定：有幻觉，第{generation_count}次，重新生成---")
        return "generate"


def route_from_answer(state):
    """
    答案质量评估后的路由：读取 check_answer 写入的结果和 not_useful_count。

    not_useful_count 由 transform_query 节点自增，此处读取的是本轮 transform 之前的值：
      - "yes"（解决了问题）→ END
      - "no"（未解决）且 not_useful_count < 2 → transform_query（后续节点会自增计数器）
      - "no"（未解决）且 not_useful_count >= 2 → not_useful_fallback（兜底）
    """
    not_useful_count = state.get("not_useful_count", 0)
    answer_result = state.get("answer_result", "no")

    if answer_result == "yes":
        log.info("---判定：生成内容准确回答问题---")
        return "useful"
    elif not_useful_count >= 2:
        log.info("---答案质量改写已达2次上限，强制结束---")
        return "transform_max_retries"
    else:
        log.info(f"---判定：答案未解决问题，第{not_useful_count}次，改写查询---")
        return "not useful"


def decide_to_generate(state):
    """
    决定是生成回答还是重新优化问题

    Args:
        state (dict): 当前图状态，包含问题和过滤后的文档

    Returns:
        str: 下一节点的名称（transform_query或generate）
    """
    log.info("---ASSESS GRADED DOCUMENTS---")  # 阶段标识

    filtered_documents = state["documents"]  # 获取已过滤文档
    transform_count = state.get("transform_count", 0)

    if not filtered_documents:  # 如果没有相关文档
        if transform_count >= 1:
            log.info("---决策：所有文档都与问题无关，已改写1次，转为web查询问题---")
            return "web_search"
        log.info("---决策：所有文档都与问题无关，将转换查询问题---")
        return "transform_query"  # 返回问题优化节点
    else:  # 如果有相关文档
        log.info("---决策：生成最终回答---")
        return "generate"  # 返回回答生成节点


def route_after_web_grade(state):
    """
    web 路径的相关性过滤之后的路由：有过阈值的候选就生成，全部被过滤掉就直接降级，
    不进 generate。不像知识库路径的 decide_to_generate 有 transform_query 重试——
    走到这里已经是"知识库确认没有、又去网上搜"的最后一步，没有更多信息源可以再退。
    """
    documents = state.get("documents", [])
    if documents:
        log.info("---web 结果有内容过了阈值，进入生成---")
        return "generate"
    log.info("---web 结果全部低于阈值，降级为未找到资料---")
    return "web_search_fallback"


def hallucination_fallback(state):
    """
    幻觉检测三次未过：不再输出模型自己判定为不可靠的答案（原来的做法是"原样保留
    +加一句警告"，但实际上多数用户会读内容、忽略警告，等于把风险转嫁给用户）。
    改为生成失败时降级为检索——直接把检索到的原始资料给用户：问答系统答不了的
    时候，至少还能当搜索引擎用。检索到的材料本身是有价值的，能走到 generate 说明
    检索已经有结果。

    这个改动不依赖"幻觉判断准不准"这个前提：如果是误判，用户拿到原始文档，没有
    损失；如果判对了，用户避开了一个编的答案，有收益。两种情况都不比原来的做法差。

    内容来源：知识库路径优先用 parent_contexts（父块，完整上下文，比小块更适合
    直接读）；web 路径没有 parent_contexts（不经过 assemble_context），退回用
    documents（web_search 已经是精排+阈值过滤后的候选，不是原始检索结果）。

    parent_contexts 每一项本身不带 source（只有 doc_id，是哈希，人读不出来），
    按 doc_id 回查 state["documents"]（小块，metadata 里有 source 相对路径）——
    这个映射一定能查到：parent_contexts 本来就是从 documents 按 parent_id/doc_id
    分组生成的（assemble_context_node.py），不会出现查不到的情况。web 路径的
    Document.metadata["source"] 本来就是 URL，直接用，不用回查。
    """
    documents = state.get("documents", [])
    parent_contexts = state.get("parent_contexts")

    blocks = []
    if parent_contexts:
        source_by_doc_id = {
            d.metadata.get("doc_id"): d.metadata.get("source")
            for d in documents
            if d.metadata.get("doc_id")
        }
        for item in parent_contexts:
            source = source_by_doc_id.get(item.get("doc_id"), "未知来源")
            blocks.append(f"【来源：{source}】\n{item['content']}")
    else:
        for doc in documents:
            source = doc.metadata.get("source", "未知来源")
            blocks.append(f"【来源：{source}】\n{doc.page_content}")

    material = "\n\n".join(blocks)
    return {
        "generation": (
            "⚠️ 系统提示：系统未能生成可靠答案（多次核查仍检测到潜在幻觉），"
            "以下是检索到的原始资料，请自行判断：\n\n" + material
        )
    }

def not_useful_fallback(state):
    return {"generation": "⚠️ 系统提示：经多次查询优化，未能找到足够相关的资料来准确回答该问题，建议换一种方式提问。"}

def web_search_fallback(state):
    return {"generation": "⚠️ 系统提示：本地知识库和网络搜索均未找到与该问题相关的资料，建议换一种方式提问。"}





# 初始化工作流图
workflow = StateGraph(GraphState)

# 定义各状态节点
workflow.add_node("web_search", web_search)  # Tavily 检索 + 超长切分 + CrossEncoder 精排
workflow.add_node("grade_web_results", grade_web_results)  # web 路径相关性阈值过滤
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("assemble_context", assemble_context)  # small-to-big：小块->父块拼装
workflow.add_node("generate", generate)
workflow.add_node("check_hallucination", check_hallucination)  # 幻觉检测（独立节点，独立 span）
workflow.add_node("check_answer", check_answer)                # 答案质量（独立节点，独立 span）
workflow.add_node("transform_query", transform_query)
workflow.add_node("hallucination_fallback", hallucination_fallback)
workflow.add_node("not_useful_fallback", not_useful_fallback)
workflow.add_node("web_search_fallback", web_search_fallback)


# 起始固定边：纯 Corrective RAG——不做检索前路由，问题一律先检索，再用 grade_documents
# 逐条判相关性。曾经加过一层"关键词路由+检索后聚合分数二次门禁"的两段式路由（见
# CLAUDE.md 二十二节），复盘后发现这层路由在检索后才生效，已经不省检索成本，剩下的
# 唯一作用是和 grade_documents 判断同一件事、粒度更粗，纯属多余，已删除。
workflow.add_edge(START, "retrieve")

# 固定边
# web_search 的结果不经过 assemble_context——它不是 Milvus 小块，没有 parent_id，
# 精排+阈值过滤后直接进 generate（D3/D8 的父子块逻辑只针对本地检索链路）。
workflow.add_edge("web_search", "grade_web_results")
workflow.add_edge("assemble_context", "generate")      # 小块->父块拼装后再生成
workflow.add_edge("generate", "check_hallucination")   # generate 后必经幻觉检测
workflow.add_edge("hallucination_fallback", END)
workflow.add_edge("not_useful_fallback", END)
workflow.add_edge("web_search_fallback", END)
workflow.add_edge("transform_query", "retrieve")

# web 路径相关性过滤后的条件分支：有内容过阈值就生成，否则直接降级，不进 generate
workflow.add_conditional_edges(
    "grade_web_results",
    route_after_web_grade,
    {
        "generate": "generate",
        "web_search_fallback": "web_search_fallback",
    },
)

# 检索完成后固定进 grade_documents——是否覆盖该问题完全交给逐条阈值过滤
# + decide_to_generate 的改写重试/降级判断，不再有检索前置或聚合分数的路由门禁。
workflow.add_edge("retrieve", "grade_documents")

# 文档评估后的条件分支
# decide_to_generate 本身语义不变（决定"要不要生成"），只是把"要生成"的目标节点
# 从 generate 改成 assemble_context——用 mapping 做这层转译，不改函数本身的返回值，
# 因为函数名和职责就是"决定路由方向"，不该关心中间多插了哪个节点。
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "assemble_context",
        "transform_query": "transform_query",
        "web_search": "web_search",
    },
)

# 幻觉检测后的条件分支
# generation_count 已由 generate 节点自增，route_from_hallucination 只读不写
workflow.add_conditional_edges(
    "check_hallucination",
    route_from_hallucination,
    {
        "check_answer": "check_answer",
        "generate": "generate",
        "hallucination_fallback": "hallucination_fallback",
    },
)

# 答案质量评估后的条件分支
# not_useful_count 由 transform_query 节点自增，route_from_answer 读到的是自增前的值
workflow.add_conditional_edges(
    "check_answer",
    route_from_answer,
    {
        "useful": END,
        "not useful": "transform_query",
        "transform_max_retries": "not_useful_fallback",
    },
)

# 编译工作流
graph = workflow.compile()

# draw_graph(graph, 'graph_rag2.png')


MOCK_USER = {"dept_id": "public"}  # K8s 文档全部是 public 权限

if __name__ == '__main__':
    with PostgresSaver.from_conn_string(POSTGRES_URI) as saver:
        saver.setup()
        pg_graph = workflow.compile(checkpointer=saver)

        # 展示最近会话
        recent = list_sessions(10)
        if recent:
            print("\n最近的会话：")
            for s in recent:
                sid_short = s["session_id"][:8]
                print(f"  {s['session_id']}  {s['created_at'][:16]}  「{s['first_question'][:30]}」")
            print()

        resume = input("继续上次对话？输入 session_id（留空则新开会话）：").strip()
        if resume:
            session_id = resume
            print(f"续接会话: {session_id}")
        else:
            session_id = str(uuid.uuid4())
            print(f"新会话 session_id: {session_id}")

        is_first_question = not bool(resume)  # 续接的会话不重复注册

        while True:
            question = input('用户：')
            if question.lower() in ['q', 'exit', 'quit']:
                print('对话结束，拜拜！')
                break

            if is_first_question:
                create_session(session_id, question)
                is_first_question = False

            ctx = TraceContext(thread_id=session_id)
            handler = TraceHandler(ctx)
            config = {
                "callbacks": [handler],
                "configurable": {"thread_id": session_id, "trace_ctx": ctx, "dept_id": MOCK_USER["dept_id"]},
            }

            # 每轮重置，防止上轮 state 污染新问题（含新增的中间评估字段）
            inputs = {
                "question": question,
                "original_question": question,
                "documents": [],
                "parent_contexts": [],
                "generation": "",
                "transform_count": 0,
                "not_useful_count": 0,
                "generation_count": 0,
                "hallucination_result": "",
                "answer_result": "",
            }
            generation = ""
            for output in pg_graph.stream(inputs, config=config):
                for key, value in output.items():
                    pprint(f"Node '{key}':")
                    if "generation" in value:
                        generation = value["generation"]
                pprint("\n---\n")

            pprint(generation)
            print(f"[trace_id: {ctx.trace_id}]")
