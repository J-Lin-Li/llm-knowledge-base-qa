import hashlib
import os
from datetime import datetime, timezone
from typing import List

from langchain_core.documents import Document

from documents.markdown_parser import MarkdownParser, assign_parent_ids
from documents.milvus_db import MilvusVectorSave
from tracing.db import (
    delete_doc_record,
    delete_parent_chunks_by_doc,
    get_doc_record,
    get_parent_chunks_by_doc,
    upsert_doc_record,
    upsert_parent_chunk,
)
from utils.log_utils import log


def upsert_document(doc_id: str, file_path: str, dept_id: str, mv: MilvusVectorSave, source_path: str = None) -> str:
    """
    幂等入库单篇文档。

    source_path: 相对 datas/md 的路径，写进每个 chunk 的 source 字段（覆盖
      parse_markdown_to_documents 默认写入的绝对路径），K8s 重名 index.md 靠这个
      区分是哪个文件，且不受运行机器的绝对路径前缀影响。不传则保留默认绝对路径
      （verify/ 下的旧调用没有 MD_DIR 上下文，不强制要求）。

    流程：
      1. 读文件内容，算 content_hash
      2. 与 document_registry 对比——hash 未变则跳过
      3. 先保存旧 chunk + 旧父块快照，再删旧 chunk + 旧父块，再插新 chunk + 新父块
      4. insert 失败时回滚（Milvus 小块和 SQLite 父块都要回滚，不能只救一半），
         不更新 SQLite 的 document_registry

    返回值: 'skipped' | 'created' | 'updated'
    """
    with open(file_path, encoding="utf-8") as f:
        raw_content = f.read()

    content_hash = hashlib.md5(raw_content.encode()).hexdigest()
    filename = os.path.basename(file_path)

    existing = get_doc_record(doc_id)
    if existing and existing["content_hash"] == content_hash:
        log.info(f"[lifecycle] doc_id={doc_id} 内容未变，跳过")
        return "skipped"

    is_update = existing is not None

    parser = MarkdownParser()
    new_chunks, new_parent_records = parser.parse_markdown_to_documents(file_path)
    for doc in new_chunks:
        doc.metadata["doc_id"] = doc_id
        doc.metadata["dept_id"] = dept_id
        if source_path is not None:
            doc.metadata["source"] = source_path
    new_chunks, new_parent_records = assign_parent_ids(new_chunks, new_parent_records, doc_id)

    old_docs: List[Document] = []
    old_parent_records: List[dict] = []
    if is_update:
        old_docs = mv.get_chunks_by_doc_id(doc_id)
        old_parent_records = get_parent_chunks_by_doc(doc_id)
        mv.delete_by_doc_id(doc_id)
        delete_parent_chunks_by_doc(doc_id)
        log.info(f"[lifecycle] 删除旧 chunk {len(old_docs)} 个，旧父块 {len(old_parent_records)} 个")

    now = datetime.now(timezone.utc).isoformat()
    try:
        mv.add_documents(new_chunks)
        for record in new_parent_records:
            upsert_parent_chunk({
                "parent_id": record["parent_id"],
                "doc_id": doc_id,
                "dept_id": dept_id,
                "title": record.get("title"),
                "content": record["content"],
                "char_count": record["char_count"],
                "seq_in_doc": record["seq"],
                "created_at": now,
                "updated_at": now,
            })
        log.info(f"[lifecycle] 插入新 chunk {len(new_chunks)} 个，新父块 {len(new_parent_records)} 个")
    except Exception as e:
        log.error(f"[lifecycle] 插入失败，开始回滚: {e}")
        if old_docs:
            mv.add_documents(old_docs)
            log.info(f"[lifecycle] 回滚完成，重插旧 chunk {len(old_docs)} 个")
        if old_parent_records:
            for record in old_parent_records:
                upsert_parent_chunk(record)
            log.info(f"[lifecycle] 回滚完成，重插旧父块 {len(old_parent_records)} 个")
        raise

    upsert_doc_record({
        "doc_id": doc_id,
        "filename": filename,
        "dept_id": dept_id,
        "content_hash": content_hash,
        "chunk_count": len(new_chunks),
        "created_at": existing["created_at"] if existing else now,
        "updated_at": now,
    })

    action = "updated" if is_update else "created"
    log.info(f"[lifecycle] doc_id={doc_id} {action}，chunk 数={len(new_chunks)}，父块数={len(new_parent_records)}")
    return action


def delete_document(doc_id: str, mv: MilvusVectorSave) -> bool:
    """删除文档的所有 chunk、父块及 SQLite 注册记录。返回是否存在并被删除。"""
    count = mv.delete_by_doc_id(doc_id)
    parent_count = delete_parent_chunks_by_doc(doc_id)
    delete_doc_record(doc_id)
    log.info(f"[lifecycle] doc_id={doc_id} 已删除，共 {count} 个 chunk，{parent_count} 个父块")
    return count > 0
