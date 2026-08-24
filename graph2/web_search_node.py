import json
import time
import uuid
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph2.retriever_node import reranker  # 复用同一个已加载的 CrossEncoder 实例，不重复加载模型
from llm_models.all_llm import web_search_tool
from tracing.db import insert_span
from utils.log_utils import log

# 单条 Tavily 结果超过这个字符数才切分，否则整条当一段。工程默认值，不是标定过的
# 数字——CrossEncoder 有效长度约 512 token，400 字符是留了安全余量的保守估计。
WEB_SPLIT_CHARS = 400

# 每条原始 Tavily 结果（不管切没切）最终只保留分数最高的这么多段。
# 定为 1，不是 2：max_results=5 时 2 段会有 10 个候选，量级和知识库路径（4个小块
# 对应的父块）不成比例，而 web 内容质量远低于官方文档，不该给它更大的 context 空间。
# 同一篇网页切出的"最相关段"和"第二相关段"往往在讲同一件事，边际信息量低；
# 真正有价值的增量应该来自不同来源（见下面 TOP-K 来源数的取舍），不是同一篇多切一段。
# 如果后续实测 context 不够用，再放宽——但要有实测依据，不是一开始就给足。
TOP_SEGMENTS_PER_RESULT = 1


def _split_if_long(content: str, max_chars: int = WEB_SPLIT_CHARS) -> list:
    """超过 max_chars 才切分；不超过就整条当一段，不强行切。"""
    if len(content) <= max_chars:
        return [content]
    return [content[i:i + max_chars] for i in range(0, len(content), max_chars)]


def web_search(state, config: RunnableConfig = None):
    """
    基于优化后的问题进行网络搜索，并对结果做和知识库路径同一套的相关性精排。

    Tavily 单条结果可能超出 CrossEncoder 有效长度（512 token），先按定长切分成
    若干段再逐段打分，避免超长文本被静默截断导致打分不准。每条原始结果只保留
    分数最高的 TOP_SEGMENTS_PER_RESULT 段，代表这条结果参与后续的知识库路径同款
    阈值过滤（grade_web_results_node.py）。

    这里只做检索+切分+打分，不做阈值过滤——过滤和知识库路径一样拆成独立节点
    （grade_web_results），理由见该文件。
    """
    log.info("---WEB SEARCH---")
    question = state["question"]

    configurable = (config or {}).get("configurable", {})
    ctx = configurable.get("trace_ctx")
    exec_idx = ctx.current_execution("web_search") if ctx else 0

    # --- retrieval span：Tavily 原始结果 ---
    retrieval_span_id = str(uuid.uuid4())
    t0 = time.monotonic()

    # web_search_tool.invoke 偶尔会返回非预期格式（观察到过整体是字符串，大概率是
    # 调用失败时的错误文本，不是正常的结果列表），也可能直接抛异常。两种情况都不
    # 让它把整个节点带崩——按 0 条结果处理，走下游已有的 web_search_fallback 降级
    # 路径（和"结果全被阈值过滤掉"是同一条路，不需要新增节点）。web_error 记录下来
    # 写进 trace，不只是静默吞掉，方便定位是不是这里出的问题。
    web_error = None
    try:
        raw_results = web_search_tool.invoke({"query": question})
    except Exception as e:
        web_error = f"web_search_tool 调用异常: {e}"
        raw_results = []

    if isinstance(raw_results, list):
        bad = [r for r in raw_results if not isinstance(r, dict)]
        if bad:
            web_error = f"结果中有 {len(bad)} 条非 dict 格式，已丢弃"
        raw_results = [r for r in raw_results if isinstance(r, dict)]
    else:
        web_error = f"web_search_tool 返回非 list: {str(raw_results)[:300]!r}"
        raw_results = []

    if web_error:
        log.warning(f"---WEB SEARCH 异常: {web_error}---")

    retrieval_ms = int((time.monotonic() - t0) * 1000)

    if ctx:
        # trace 只存标识符和度量（url、条数、耗时），不存网页正文——原则和权限边界见
        # retriever_node.py 同名注释；question 保留的例外也一样，不在这里重复。
        insert_span({
            "trace_id": ctx.trace_id,
            "span_id": retrieval_span_id,
            "parent_span_id": None,
            "thread_id": ctx.thread_id,
            "node_name": "web_search",
            "span_type": "retrieval",
            "model_name": None,
            "input": json.dumps({"question": question}, ensure_ascii=False),
            "output": json.dumps({
                "count": len(raw_results),
                "urls": [r.get("url") for r in raw_results],
            }, ensure_ascii=False),
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_ms": retrieval_ms,
            "status": "error" if web_error else "ok",
            "error_msg": web_error,
            "extra": json.dumps({"source": "tavily"}, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_index": exec_idx,
        })

    documents = []
    if raw_results:
        # 每条原始结果独立切分、独立精排、各自只留 top-N 段
        t1 = time.monotonic()
        for result in raw_results:
            content = result.get("content", "")
            if not content:
                continue
            segments = _split_if_long(content)
            pairs = [(question, seg) for seg in segments]
            scores = reranker.predict(pairs)
            scored_segments = sorted(zip(scores, segments), key=lambda x: x[0], reverse=True)
            for score, seg in scored_segments[:TOP_SEGMENTS_PER_RESULT]:
                documents.append(Document(
                    page_content=seg,
                    metadata={"reranker_score": float(score), "source": result.get("url")},
                ))
        reranker_ms = int((time.monotonic() - t1) * 1000)

        if ctx:
            insert_span({
                "trace_id": ctx.trace_id,
                "span_id": str(uuid.uuid4()),
                "parent_span_id": retrieval_span_id,
                "thread_id": ctx.thread_id,
                "node_name": "web_search",
                "span_type": "reranker",
                "model_name": "bge-reranker-base",
                "input": json.dumps({"question": question, "raw_result_count": len(raw_results)}, ensure_ascii=False),
                "output": json.dumps({
                    "kept": len(documents),
                    "ranking": [
                        {"score": round(d.metadata["reranker_score"], 4), "source": d.metadata.get("source")}
                        for d in documents
                    ],
                }, ensure_ascii=False),
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_ms": reranker_ms,
                "status": "ok",
                "error_msg": None,
                "extra": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "execution_index": exec_idx,
            })

    return {"documents": documents, "question": question}
