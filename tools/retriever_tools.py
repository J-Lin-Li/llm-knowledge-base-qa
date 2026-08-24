from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from pymilvus import AnnSearchRequest, RRFRanker

from documents.milvus_db import MilvusVectorSave
from llm_models.embeddings_model import bge_embedding

mv = MilvusVectorSave()
mv.create_connection()

# 保留原有 retriever 供 Graph 1 使用（filter 对混合检索不生效，已知问题）
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',
    search_kwargs={
        "k": 10,
        "score_threshold": 0.1,
        "ranker_type": "rrf",
        "ranker_params": {"k": 100},
        'filter': {"category": "content"}
    }
)

retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    '搜索并返回关于 Milvus 向量数据库的信息，内容涵盖：基本概念、架构介绍、部署、配置、性能调优、FAQ 和故障排查'
)

# ── 权限感知混合检索 ──────────────────────────────────────────────────────────

ALLOWED_DEPTS = {"engineering", "ops", "product", "public"}

_OUTPUT_FIELDS = [
    "text", "category", "source", "filename", "filetype",
    "title", "category_depth", "dept_id", "doc_id",
    "parent_id", "chunk_offset", "section_seq",
]


def build_filter_expr(dept_id: str) -> str:
    """
    白名单校验后构造 Milvus filter 表达式，防止表达式注入。

    不再过滤 category == 'content'：入库判据已经从"按元素类型"改成"按有无实质
    内容"（见 markdown_parser.py），库里现在只有通过内容判据筛进来的真实内容，
    category 字段仅保留作追踪/调试用的元数据，不参与检索过滤。
    """
    if dept_id not in ALLOWED_DEPTS:
        raise ValueError(f"dept_id '{dept_id}' 不在允许列表内: {ALLOWED_DEPTS}")
    return f"dept_id == '{dept_id}'"


def auth_hybrid_search(question: str, dept_id: str, recall_k: int = 50, fuse_k: int = 20):
    """
    绕过 as_retriever，直接调 pymilvus hybrid_search。
    expr 注入两个 AnnSearchRequest，确保 dense 和 sparse 两路都做预过滤。
    recall_k：dense/sparse 单路各自的候选数量上限（HNSW ef 必须 >= 这个值，随其联动）。
    fuse_k：RRF 融合后的最终截断数量，即交给下游 CrossEncoder 精排的候选池大小。
    """
    expr = build_filter_expr(dept_id)
    dense_vec = bge_embedding.embed_query(question)

    dense_req = AnnSearchRequest(
        data=[dense_vec],
        anns_field="dense",
        param={"metric_type": "IP", "params": {"ef": recall_k}},
        limit=recall_k,
        expr=expr,
    )
    sparse_req = AnnSearchRequest(
        data=[question],
        anns_field="sparse",
        param={"metric_type": "BM25", "params": {"drop_ratio_build": 0.2}},
        limit=recall_k,
        expr=expr,
    )

    results = mv.vector_store_saved.col.hybrid_search(
        [dense_req, sparse_req],
        RRFRanker(k=60),
        limit=fuse_k,
        output_fields=_OUTPUT_FIELDS,
    )

    docs = []
    for hit in results[0]:
        metadata = {f: hit.fields.get(f) for f in _OUTPUT_FIELDS if f != "text"}
        metadata["id"] = hit.id  # 主键不在 output_fields 里，hit 对象自带；D8 追踪要用
        docs.append(Document(
            page_content=hit.fields.get("text", ""),
            metadata=metadata,
        ))
    return docs


def get_sibling_chunks(parent_id: str, dept_id: str) -> list[dict]:
    """
    按 parent_id 查询该父块下的所有小块，按 chunk_offset 排序返回。
    用于父块超过 PARENT_UPPER_BOUND 时的退化路径（命中小块 + 相邻若干小块）。
    dept_id 参与 expr 过滤，和 auth_hybrid_search 同样的权限前置过滤原则。
    """
    if dept_id not in ALLOWED_DEPTS:
        raise ValueError(f"dept_id '{dept_id}' 不在允许列表内: {ALLOWED_DEPTS}")
    expr = f"parent_id == '{parent_id}' and dept_id == '{dept_id}'"
    rows = mv.vector_store_saved.col.query(
        expr=expr,
        output_fields=["text", "chunk_offset"],
        consistency_level="Strong",
    )
    rows.sort(key=lambda r: r.get("chunk_offset") or 0)
    return rows


def get_chunks_by_doc_id_for_dept(doc_id: str, dept_id: str) -> list[dict]:
    """
    trace 回查用：trace span 只存 doc_id/parent_id 等标识符，不存 chunk 原文
    （见 graph2/retriever_node.py 的说明）；需要看原文时按 doc_id 走这里取。

    复用 build_filter_expr 做 dept_id 白名单校验和 expr 拼接，和 auth_hybrid_search/
    get_sibling_chunks 走同一套权限前置过滤，不单独写一套——避免两处校验逻辑各自
    维护、早晚漂移出不一致（一边改了权限规则另一边没跟着改，又是一次静默失效）。

    不要和 documents/milvus_db.py 的 MilvusVectorSave.get_chunks_by_doc_id 混用：
    那个是 lifecycle 删除前做快照回滚用的内部函数，只按 doc_id 过滤、不校验
    dept_id，是可信上下文（内部管理脚本）专用，不能拿来给 trace 回查这种可能被
    任意调用方触发的场景用——会绕开部门权限。这里函数名特意带 _for_dept 后缀，
    提醒调用方这是过权限校验的版本。

    文档被删除后查不到内容是预期行为，不做任何"缓存原文防止查不到"的补偿——
    文档删了，它的内容就不该还能通过 trace 看到。
    """
    expr = f"{build_filter_expr(dept_id)} and doc_id == '{doc_id}'"
    rows = mv.vector_store_saved.col.query(
        expr=expr,
        output_fields=_OUTPUT_FIELDS,
        consistency_level="Strong",
    )
    rows.sort(key=lambda r: r.get("chunk_offset") or 0)
    return rows
