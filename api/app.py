"""
RAG 知识库 API 服务

启动：
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

接口：
    POST /chat          发起问答
    GET  /health        健康检查

生产快速路径（默认）：
    enable_hallucination_check=false, enable_answer_check=false
    → 跳过阻塞式 LLM 质量门禁，直接返回答案
    → 质量检查在 BackgroundTasks 里异步跑，结果写入 tracing.db（结构化、可查询）

完整闭环路径（CLI / 调试）：
    enable_hallucination_check=true, enable_answer_check=true
    → 与 graph2/graph_2.py __main__ 行为一致
"""

import asyncio
import json
import time
import uuid
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from pydantic import BaseModel

from graph2.graph_2 import workflow
from graph2.grade_hallucinations_chain import hallucination_grader_chain
from graph2.grade_answer_chain import answer_grader_chain
from tracing.context import TraceContext
from tracing.handler import TraceHandler
from tracing.db import create_session, insert_span
from utils.env_utils import POSTGRES_URI, BG_QUALITY_SAMPLE_RATE
from utils.log_utils import log

# 同步 RAG 链路跑在线程池里，FastAPI 事件循环不被阻塞
_executor = ThreadPoolExecutor(max_workers=4)
_api_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _api_graph
    with PostgresSaver.from_conn_string(POSTGRES_URI) as saver:
        saver.setup()
        _api_graph = workflow.compile(checkpointer=saver)
        log.info("RAG graph 初始化完成，服务就绪")
        yield
    log.info("服务关闭，Postgres 连接已释放")


app = FastAPI(title="RAG 企业知识库 API", lifespan=lifespan)


# ── 请求 / 响应模型 ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    dept_id: str = "public"
    # 生产快速路径默认关闭质量门禁，由后台异步监控
    enable_hallucination_check: bool = False
    enable_answer_check: bool = False


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    trace_id: str


# ── 核心执行（同步，在线程池里跑）────────────────────────────────────────────

def _run_graph(question: str, session_id: str, dept_id: str,
               enable_h: bool, enable_a: bool) -> tuple[dict, str]:
    """在线程池内同步执行 RAG 图，返回 (final_state, trace_id)。"""
    ctx = TraceContext(thread_id=session_id)
    handler = TraceHandler(ctx)

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
    config = {
        "callbacks": [handler],
        "configurable": {
            "thread_id": session_id,
            "trace_ctx": ctx,
            "dept_id": dept_id,
            "enable_hallucination_check": enable_h,
            "enable_answer_check": enable_a,
        },
    }
    final_state = _api_graph.invoke(inputs, config=config)
    return final_state, ctx.trace_id


# ── 后台质量监控 ───────────────────────────────────────────────────────────────

def _bg_quality_check(answer: str, question: str, documents: list, trace_id: str, thread_id: str) -> None:
    """跳过阻塞检测时，在后台异步跑幻觉检测 + 答案质量，结果写入 tracing.db（结构化、
    可按 trace_id/时间查询），不再只写应用日志——log 是即时可读但不可聚合分析，
    真要看"过去一周幻觉率"这类趋势，得有结构化存储才行。

    span 只存文档 doc_id/数量，不存原文，和 retriever_node.py 等其余节点的 trace
    原则一致（tracing.db 是单文件 SQLite，没有 Milvus 那套 dept_id 前置过滤）。

    parent_span_id 给 None：这次检查发生在原始请求已经返回之后，不是嵌套在
    那次请求实时的调用栈里，没有一个"正在进行中"的父 span 可挂。
    """
    span_id = str(uuid.uuid4())
    t0 = time.monotonic()
    output = None
    error_msg = None
    try:
        h = hallucination_grader_chain.invoke({"documents": documents, "generation": answer})
        a = answer_grader_chain.invoke({"question": question, "generation": answer})
        output = {"hallucination": h.binary_score, "answer_quality": a.binary_score}
        log.info(
            f"[BG质检] trace={trace_id[:8]} "
            f"幻觉={h.binary_score} 答案质量={a.binary_score}"
        )
    except Exception as e:
        error_msg = str(e)
        log.warning(f"[BG质检] 失败 trace={trace_id[:8]}: {e}")

    insert_span({
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "thread_id": thread_id,
        "node_name": "bg_quality_check",
        "span_type": "node",
        "model_name": "deepseek-chat",
        "input": json.dumps({
            "question": question,
            "doc_count": len(documents),
            "doc_ids": [d.metadata.get("doc_id") for d in documents],
        }, ensure_ascii=False),
        "output": json.dumps(output, ensure_ascii=False) if output else None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "status": "error" if error_msg else "ok",
        "error_msg": error_msg,
        "extra": json.dumps({"source": "background_async_quality_check"}, ensure_ascii=False),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_index": 0,
    })


# ── 接口 ───────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    session_id = request.session_id or str(uuid.uuid4())

    try:
        loop = asyncio.get_event_loop()
        final_state, trace_id = await loop.run_in_executor(
            _executor,
            lambda: _run_graph(
                request.question,
                session_id,
                request.dept_id,
                request.enable_hallucination_check,
                request.enable_answer_check,
            ),
        )
    except Exception as e:
        log.error(f"RAG pipeline 执行失败: {e}")
        raise HTTPException(status_code=500, detail="RAG pipeline failed")

    answer = final_state.get("generation", "")

    # 快速路径跳过了质量门禁时，按采样率决定是否后台异步补跑监控
    if (not request.enable_hallucination_check or not request.enable_answer_check) \
            and random.random() < BG_QUALITY_SAMPLE_RATE:
        background_tasks.add_task(
            _bg_quality_check,
            answer,
            request.question,
            final_state.get("documents", []),
            trace_id,
            session_id,
        )

    return ChatResponse(answer=answer, session_id=session_id, trace_id=trace_id)


@app.get("/health")
async def health():
    return {"status": "ok", "graph_ready": _api_graph is not None}
