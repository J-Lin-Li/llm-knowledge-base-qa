import json
import time
import uuid
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from graph2.grade_documents_node import filter_by_reranker_score
from tracing.db import insert_span
from utils.log_utils import log

# 独立常量，不复用 RERANKER_THRESHOLD（知识库路径）。应该比它更松：知识库是官方
# 文档，表述规范、术语准确，同一个问题下精排分数天然更高；网页是各种博客，表述松散，
# 用知识库那套标准会把 web 结果过滤得过狠。而且误判代价不对称——走到这里说明知识库
# 已经确认没有，这时候"有点相关的网页内容"比"什么都没有"强。
# 具体值先留空：等 RERANKER_THRESHOLD 的 0.1/0.3 标定实验跑完再回填，不能没有依据地
# 拍一个数字当结论用。未标定时使用会显式报错，不会静默跑。
WEB_RERANKER_THRESHOLD = 0.1


def grade_web_results(state, config: RunnableConfig = None):
    """
    web 路径的相关性阈值过滤，和知识库路径的 grade_documents 是同一套阈值比较逻辑
    （filter_by_reranker_score，定义在 grade_documents_node.py，两边共用避免重复
    写一遍循环），只是换了独立的阈值常量和 node_name，分开注册、分开写 trace span。
    """
    log.info("---CHECK WEB RESULT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    configurable = (config or {}).get("configurable", {})
    ctx = configurable.get("trace_ctx")
    exec_idx = ctx.current_execution("grade_web_results") if ctx else 0

    if WEB_RERANKER_THRESHOLD is None:
        raise RuntimeError(
            "WEB_RERANKER_THRESHOLD 尚未标定，无法对 web 检索结果做相关性过滤。"
            "需要先跑完 RERANKER_THRESHOLD 的 0.1/0.3 对比实验，再回填这个值。"
        )

    t0 = time.monotonic()
    filtered_docs = filter_by_reranker_score(documents, WEB_RERANKER_THRESHOLD)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    if ctx:
        insert_span({
            "trace_id": ctx.trace_id,
            "span_id": str(uuid.uuid4()),
            "parent_span_id": None,
            "thread_id": ctx.thread_id,
            "node_name": "grade_web_results",
            "span_type": "node",
            "model_name": None,
            "input": json.dumps({"question": question, "candidate_count": len(documents)}, ensure_ascii=False),
            "output": json.dumps({"kept": len(filtered_docs), "threshold": WEB_RERANKER_THRESHOLD}, ensure_ascii=False),
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_ms": elapsed_ms,
            "status": "ok",
            "error_msg": None,
            "extra": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_index": exec_idx,
        })

    return {"documents": filtered_docs, "question": question}
