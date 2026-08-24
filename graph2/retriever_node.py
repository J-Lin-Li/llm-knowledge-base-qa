import json
import time
import uuid
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig
from sentence_transformers import CrossEncoder

from tools.retriever_tools import auth_hybrid_search
from tracing.db import insert_span
from utils.log_utils import log

reranker = CrossEncoder('BAAI/bge-reranker-base')


def retrieve(state, config: RunnableConfig = None):
    log.info("---去知识库中检索文档---")
    question = state["question"]

    configurable = (config or {}).get("configurable", {})
    ctx = configurable.get("trace_ctx")
    dept_id = configurable.get("dept_id", "public")
    exec_idx = ctx.current_execution("retrieve") if ctx else 0

    # --- retrieval span: 包裹权限感知混合检索 ---
    retrieval_span_id = str(uuid.uuid4())
    t0 = time.monotonic()
    documents = auth_hybrid_search(question, dept_id)
    retrieval_ms = int((time.monotonic() - t0) * 1000)

    if ctx:
        # trace 只存标识符和度量，不存 chunk 原文——tracing.db 是单文件 SQLite，没有
        # Milvus 那套 dept_id 前置过滤，原文一旦写进去，谁能读这个文件就能看到全部
        # 部门的内容，且观测数据的留存周期（通常数月）和权限的实时性错配（人调离部门
        # 后旧 trace 里的内容还在）。需要看原文时按 doc_id 回查
        # （tools/retriever_tools.get_chunks_by_doc_id_for_dept），复用检索链路本身
        # 的 dept_id 权限校验，不在 trace 里留一份没有权限隔离的副本。
        #
        # question 是例外，继续原样存：它是用户输入不是知识库内容，权限模型不适用于它；
        # 且没有问题的 trace 基本没法读（看到"命中 doc_A 分数 0.8"却不知道在问什么，
        # 等于没有）。但这个例外有边界——用户问题本身也可能包含敏感信息，更严格的生产
        # 环境会对问题做脱敏或哈希后再入库，这里权衡后选择不做，如果之后合规要求收紧，
        # 这是需要重新评估的点。
        insert_span({
            "trace_id": ctx.trace_id,
            "span_id": retrieval_span_id,
            "parent_span_id": None,
            "thread_id": ctx.thread_id,
            "node_name": "retrieve",
            "span_type": "retrieval",
            "model_name": None,
            "input": json.dumps({"question": question, "dept_id": dept_id}, ensure_ascii=False),
            "output": json.dumps({
                "count": len(documents),
                "docs": [doc.metadata for doc in documents],
            }, ensure_ascii=False),
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_ms": retrieval_ms,
            "status": "ok",
            "error_msg": None,
            "extra": json.dumps({"source": "milvus_hybrid", "dept_id": dept_id}, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_index": exec_idx,
        })

    if documents:
        pairs = [(question, doc.page_content) for doc in documents]

        # --- reranker span: 包裹 CrossEncoder 重排 ---
        t1 = time.monotonic()
        scores = reranker.predict(pairs)
        reranker_ms = int((time.monotonic() - t1) * 1000)

        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        for score, doc in scored_docs:
            doc.metadata["reranker_score"] = float(score)
        documents = [doc for _, doc in scored_docs[:4]]
        log.info(f"---Reranker 重排后保留 {len(documents)} 个文档---")

        if ctx:
            insert_span({
                "trace_id": ctx.trace_id,
                "span_id": str(uuid.uuid4()),
                "parent_span_id": retrieval_span_id,
                "thread_id": ctx.thread_id,
                "node_name": "retrieve",
                "span_type": "reranker",
                "model_name": "bge-reranker-base",
                "input": json.dumps({"question": question, "doc_count": len(pairs)}, ensure_ascii=False),
                "output": json.dumps({
                    "kept": len(documents),
                    "ranking": [
                        {
                            "score": round(float(s), 4),
                            "doc_id": doc.metadata.get("doc_id"),
                            "chunk_id": doc.metadata.get("id"),
                            "source": doc.metadata.get("source"),
                        }
                        for s, doc in scored_docs
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
