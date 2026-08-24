import json
import time
import uuid
from datetime import datetime, timezone

from langchain_core.runnables import RunnableConfig

from tools.retriever_tools import get_sibling_chunks
from tracing.db import get_parent_chunk, insert_span
from utils.log_utils import log

# 必须和 documents/markdown_parser.py 里的同名常量保持一致——两处独立定义是因为
# 职责不同（parser 端用于离线判断父块要不要建，这里用于在线运行时决定要不要退化），
# 但数值必须对齐，否则"父块该退化却没退化"或反过来。
PARENT_UPPER_BOUND = 2000

# 退化路径：命中位置前后各取多少个相邻小块。工程默认值，不是用户明确决策，可调。
NEIGHBOR_WINDOW = 1


def assemble_context(state, config: RunnableConfig = None):
    """
    独立节点，在 grade_documents（阈值过滤）之后、generate 之前执行。

    职责：把命中的小块换成父块内容（或因父块超长而退化为"命中小块+相邻若干"），
    按 parent_id 去重，按 D7 规则排序，写入 state["parent_contexts"] 供 generate 使用。

    为什么是独立节点，不塞进 grade_documents 或 generate：
      - 追踪归属：塞进 generate 会让 SQLite 查询/去重的 span 混入 generate 名下，
        重蹈十四节把幻觉检测/答案评估从条件边函数拆出来之前的覆辙
      - 职责边界：generate 只负责"把 context 交给 LLM"，不该同时决定 context 是什么
      - 可旁路：独立节点意味着这一层能整体关闭，图结构不用变
    """
    documents = state["documents"]
    configurable = (config or {}).get("configurable", {})
    ctx = configurable.get("trace_ctx")
    dept_id = configurable.get("dept_id", "public")
    exec_idx = ctx.current_execution("assemble_context") if ctx else 0

    t0 = time.monotonic()

    # ── Step 1：按 parent_id 分组；parent_id 为空（章节切不动）的小块单独处理 ──────
    parent_groups: dict = {}
    standalone_items = []

    hit_chunks_log = []       # D8：阈值过滤后进入合并的小块 id + 分数
    chunk_to_parent_log = []  # D8：每个小块 -> parent_id 的映射

    for doc in documents:
        pid = doc.metadata.get("parent_id")
        score = doc.metadata.get("reranker_score")
        chunk_id = doc.metadata.get("id")
        hit_chunks_log.append({"chunk_id": chunk_id, "score": score})
        chunk_to_parent_log.append({"chunk_id": chunk_id, "parent_id": pid})

        if pid is None:
            standalone_items.append({
                "doc_id": doc.metadata.get("doc_id"),
                "section_seq": doc.metadata.get("section_seq") or 0,
                "content": doc.page_content,
                "score": score if score is not None else 0.0,
                "parent_id": None,
                "char_count": len(doc.page_content),
                "degraded": False,
            })
        else:
            g = parent_groups.setdefault(pid, {"chunks": [], "max_score": None})
            g["chunks"].append(doc)
            if score is not None:
                g["max_score"] = score if g["max_score"] is None else max(g["max_score"], score)

    # ── Step 2：查父块内容，D6 二次权限校验，超上界退化 ──────────────────────────
    parent_items = []
    parent_lengths_log = {}   # D8：每个父块长度
    hit_positions_log = {}    # D8：命中小块在父块中的位置

    for pid, g in parent_groups.items():
        record = get_parent_chunk(pid)
        if record is None:
            # 理论上不该发生（parent_id 存在但 SQLite 查不到），兜底退化为只用命中小块
            log.error(f"[assemble_context] parent_id={pid} 在 SQLite 中查不到，退化为仅用命中小块")
            content = "\n".join(d.page_content for d in g["chunks"])
            first = g["chunks"][0]
            parent_items.append({
                "doc_id": first.metadata.get("doc_id"),
                "section_seq": first.metadata.get("section_seq") or 0,
                "content": content,
                "score": g["max_score"] or 0.0,
                "parent_id": pid,
                "char_count": len(content),
                "degraded": True,
            })
            continue

        # D6：二次权限校验。纵深防御，不是修复越权漏洞——当前 dept_from_rel_path
        # 按文档目录分类，同一篇文档的所有小块/父块 dept_id 理论上一致，父块从不
        # 会混不同权限的小块。这里验证的是这道防线本身没写错，不是发现并堵上了
        # 一个真实能被触发的越权路径。
        if record["dept_id"] != dept_id:
            log.error(f"[assemble_context] 权限校验失败：parent_id={pid} dept={record['dept_id']} "
                      f"但当前请求 dept={dept_id}，丢弃该父块")
            continue

        hit_offsets = sorted(
            d.metadata.get("chunk_offset") for d in g["chunks"] if d.metadata.get("chunk_offset") is not None
        )
        hit_positions_log[pid] = hit_offsets
        parent_lengths_log[pid] = record["char_count"]

        if record["char_count"] <= PARENT_UPPER_BOUND:
            parent_items.append({
                "doc_id": record["doc_id"],
                "section_seq": record["seq_in_doc"],
                "content": record["content"],
                "score": g["max_score"] or 0.0,
                "parent_id": pid,
                "char_count": record["char_count"],
                "degraded": False,
            })
        else:
            # 超上界：退化为"命中小块 + 前后相邻若干小块"，不返回整章
            siblings = get_sibling_chunks(pid, dept_id)
            keep = set()
            for off in hit_offsets:
                for delta in range(-NEIGHBOR_WINDOW, NEIGHBOR_WINDOW + 1):
                    keep.add(off + delta)
            selected = [s for s in siblings if (s.get("chunk_offset") or 0) in keep]
            content = "\n".join(s["text"] for s in selected) if selected else \
                "\n".join(d.page_content for d in g["chunks"])
            parent_items.append({
                "doc_id": record["doc_id"],
                "section_seq": record["seq_in_doc"],
                "content": content,
                "score": g["max_score"] or 0.0,
                "parent_id": pid,
                "char_count": len(content),
                "degraded": True,
            })

    all_items = parent_items + standalone_items

    # ── Step 3：D7 排序 —— 同文档内按 section_seq（章节原序）排，跨文档按该文档最高分排序 ──
    doc_groups: dict = {}
    for item in all_items:
        doc_groups.setdefault(item["doc_id"], []).append(item)

    doc_order = sorted(
        doc_groups.keys(),
        key=lambda d: max(it["score"] for it in doc_groups[d]),
        reverse=True,
    )

    final_contexts = []
    for d in doc_order:
        group_sorted = sorted(doc_groups[d], key=lambda it: it["section_seq"])
        final_contexts.extend(group_sorted)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ── D8：手动埋点 —— 不是 LangChain Runnable，callback 到不了（和 CrossEncoder/
    # 混合检索同理）。父块全文不进 trace，只存 parent_id，需要时按 id 回查（延续
    # 十节 D1 原则：retrieval span 存全文，下游 span 只存引用）。
    if ctx:
        insert_span({
            "trace_id": ctx.trace_id,
            "span_id": str(uuid.uuid4()),
            "parent_span_id": None,
            "thread_id": ctx.thread_id,
            "node_name": "assemble_context",
            "span_type": "node",
            "model_name": None,
            "input": json.dumps({
                "hit_chunks": hit_chunks_log,
                "chunk_to_parent": chunk_to_parent_log,
            }, ensure_ascii=False),
            "output": json.dumps({
                "parent_count_after_dedup": len(all_items),
                "parent_lengths": parent_lengths_log,
                "hit_positions_in_parent": hit_positions_log,
                "final_order": [
                    {"parent_id": it["parent_id"], "doc_id": it["doc_id"], "degraded": it["degraded"]}
                    for it in final_contexts
                ],
            }, ensure_ascii=False),
            "prompt_tokens": None,
            "completion_tokens": None,
            "latency_ms": elapsed_ms,
            "status": "ok",
            "error_msg": None,
            "extra": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "execution_index": exec_idx,
        })

    return {"parent_contexts": final_contexts}
