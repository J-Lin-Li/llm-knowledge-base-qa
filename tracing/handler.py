import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tracing.context import TraceContext
from tracing.db import insert_span, update_span


class TraceHandler(BaseCallbackHandler):
    """Records one span per LLM call to SQLite.

    grade_documents gets its own 'node'-type span (question in, docs_kept out,
    latency) via the on_chain_start/on_chain_end special case below, regardless
    of what happens inside it. It used to also correlate up to 4 child 'llm'
    spans back when it called an LLM grader per document (2026-07/08); since
    2026-08-04 it's a pure reranker-score threshold comparison with zero LLM
    calls, so that correlation logic (_chain_to_node / _grade_doc_seq / the
    on_llm_start branch) is commented out as dead code below, not deleted.
    All other LLM-calling nodes still get one 'llm' span per call.

    Run-id hierarchy used for parent resolution:
        graph (parent=None)
          └─ node run (parent=graph_run_id)
               └─ chain run (parent=node_run_id)
                    └─ llm run  (parent=chain_run_id)
    """

    def __init__(self, ctx: TraceContext):
        super().__init__()
        self.ctx = ctx
        self._graph_run_id: Optional[str] = None
        # node_run_id -> {node_name, span_id, start_time, exec_idx}
        self._node_runs: dict[str, dict] = {}
        # llm_run_id -> {start_time, node_name, model_name, parent_span_id, doc_seq}
        self._pending_llm: dict[str, dict] = {}
        # 2026-08-20 注释停用（不删）：grade_documents 降级成纯阈值比较（不再调 LLM）
        # 之后，下面两个字典和它们的消费逻辑（on_chain_start 的 sub-chain 映射块、
        # on_llm_start 的 grade_documents 分支、on_chain_end 的 pop）永远不会被触发——
        # 没有 LLM 子调用就不会有 on_llm_start(node_name="grade_documents")，也就
        # 用不上"关联到哪个父 span、第几个文档"这套簿记。grade_documents 的节点级
        # span（question/docs_kept/latency）走的是 on_chain_start/on_chain_end 里
        # 另一段判断（`if node_name == "grade_documents":` 那段创建/更新 span 的代码），
        # 那段仍在正常工作，没有一起停用。
        # # chain_run_id -> node_run_id  (only tracked for grade_documents)
        # self._chain_to_node: dict[str, str] = {}
        # # node_run_id -> doc sequence counter (how many LLM calls so far in this round)
        # self._grade_doc_seq: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Chain events: detect graph root and node boundaries
    # ------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        run_id_s = str(run_id)
        parent_s = str(parent_run_id) if parent_run_id else None
        node_name = (metadata or {}).get("langgraph_node")

        if parent_run_id is None:
            # The CompiledGraph itself
            self._graph_run_id = run_id_s
            return

        if parent_s == self._graph_run_id and node_name:
            # Direct child of graph = a node execution
            exec_idx = self.ctx.next_execution(node_name)
            node_info = {
                "node_name": node_name,
                "span_id": str(uuid.uuid4()),
                "start_time": time.monotonic(),
                "exec_idx": exec_idx,
            }
            self._node_runs[run_id_s] = node_info

            if node_name == "grade_documents":
                # self._grade_doc_seq[run_id_s] = 0  # 2026-08-20 停用，见 __init__ 注释
                insert_span({
                    "trace_id": self.ctx.trace_id,
                    "span_id": node_info["span_id"],
                    "parent_span_id": None,
                    "thread_id": self.ctx.thread_id,
                    "node_name": "grade_documents",
                    "span_type": "node",
                    "model_name": None,
                    "input": json.dumps(
                        {"question": inputs.get("question", "")},
                        ensure_ascii=False,
                    ),
                    "output": None,
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "latency_ms": None,
                    "status": "running",
                    "error_msg": None,
                    "extra": json.dumps({"round": exec_idx}, ensure_ascii=False),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "execution_index": exec_idx,
                })
            return

        # 2026-08-20 停用（不删），见 __init__ 注释：grade_documents 内部不再有任何
        # 子链调用，这段永远等不到匹配对象。
        # # Sub-chain inside a node: track chain → node mapping for grade_documents
        # if parent_s in self._node_runs:
        #     ninfo = self._node_runs[parent_s]
        #     if ninfo["node_name"] == "grade_documents":
        #         self._chain_to_node[run_id_s] = parent_s

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_id_s = str(run_id)
        node_info = self._node_runs.pop(run_id_s, None)
        if node_info and node_info["node_name"] == "grade_documents":
            latency_ms = int((time.monotonic() - node_info["start_time"]) * 1000)
            kept = len(outputs.get("documents", []))
            update_span(
                node_info["span_id"],
                latency_ms=latency_ms,
                status="ok",
                output=json.dumps({"docs_kept": kept}, ensure_ascii=False),
            )
            # self._grade_doc_seq.pop(run_id_s, None)  # 2026-08-20 停用，见 __init__ 注释

    # ------------------------------------------------------------------
    # LLM events: one span per LLM call
    # ------------------------------------------------------------------
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        parent_s = str(parent_run_id) if parent_run_id else None
        node_name = (metadata or {}).get("langgraph_node", "unknown")

        parent_span_id: Optional[str] = None
        doc_seq: Optional[int] = None

        # 2026-08-20 停用（不删），见 __init__ 注释：grade_documents 不再调 LLM，
        # on_llm_start 永远不会以 node_name=="grade_documents" 的形式触发，这段死代码。
        # if node_name == "grade_documents" and parent_s:
        #     node_run_id = self._chain_to_node.get(parent_s)
        #     if node_run_id and node_run_id in self._node_runs:
        #         parent_span_id = self._node_runs[node_run_id]["span_id"]
        #         doc_seq = self._grade_doc_seq.get(node_run_id, 0)
        #         self._grade_doc_seq[node_run_id] = doc_seq + 1

        self._pending_llm[str(run_id)] = {
            "start_time": time.monotonic(),
            "node_name": node_name,
            "model_name": serialized.get("name", "unknown"),
            "parent_span_id": parent_span_id,
            "doc_seq": doc_seq,
        }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_id_s = str(run_id)
        pending = self._pending_llm.pop(run_id_s, None)
        if not pending:
            return

        latency_ms = int((time.monotonic() - pending["start_time"]) * 1000)
        node_name = pending["node_name"]
        usage = (response.llm_output or {}).get("token_usage", {})

        output_text = ""
        if response.generations:
            gen = response.generations[0][0]
            output_text = getattr(gen, "text", "") or str(gen)

        # execution_index: read from the currently active node run for this node
        exec_index = next(
            (v["exec_idx"] for v in self._node_runs.values() if v["node_name"] == node_name),
            0,
        )

        extra: dict = {
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens", 0),
        }
        if pending["doc_seq"] is not None:
            extra["doc_seq"] = pending["doc_seq"]

        insert_span({
            "trace_id": self.ctx.trace_id,
            "span_id": run_id_s,
            "parent_span_id": pending["parent_span_id"],
            "thread_id": self.ctx.thread_id,
            "node_name": node_name,
            "span_type": "llm",
            "model_name": pending["model_name"],
            "input": json.dumps({"node": node_name}, ensure_ascii=False),
            "output": json.dumps({"text": output_text[:1000]}, ensure_ascii=False),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "latency_ms": latency_ms,
            "status": "ok",
            "error_msg": None,
            "extra": json.dumps(extra, ensure_ascii=False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_index": exec_index,
        })

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        run_id_s = str(run_id)
        pending = self._pending_llm.pop(run_id_s, None)
        if not pending:
            return
        latency_ms = int((time.monotonic() - pending["start_time"]) * 1000)

        insert_span({
            "trace_id": self.ctx.trace_id,
            "span_id": run_id_s,
            "parent_span_id": pending.get("parent_span_id"),
            "thread_id": self.ctx.thread_id,
            "node_name": pending["node_name"],
            "span_type": "llm",
            "model_name": pending["model_name"],
            "input": None,
            "output": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_ms": latency_ms,
            "status": "error",
            "error_msg": str(error),
            "extra": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_index": 0,
        })
