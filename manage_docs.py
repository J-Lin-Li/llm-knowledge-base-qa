"""
知识库文档管理脚本
==================

用法（在项目根目录下，激活 rag_project 环境后运行）：

  查看所有已注册文档：
    python manage_docs.py list

  查看某篇文档的详细状态：
    python manage_docs.py status <doc_id>
    例：python manage_docs.py status 73654fb858684160

  新增或更新一篇文档（幂等，内容未变则自动跳过）：
    python manage_docs.py update <md文件路径> [--dept <dept_id>]
    例：python manage_docs.py update datas/md/tasks/debug/troubleshoot-clusters.md
    例：python manage_docs.py update datas/md/concepts/security/xxx.md --dept engineering

    dept_id 可选值：engineering / ops / product / public
    不传 --dept 时自动按目录映射（如 datas/md/concepts/security/xxx.md → engineering），
    映射表见 documents/write_milvus.py 的 DIR_TO_DEPT。目录不在映射表里默认归 public 部门。

  删除一篇文档（同时清除 Milvus chunk 和注册记录）：
    python manage_docs.py delete <doc_id>
    例：python manage_docs.py delete 73654fb858684160

    doc_id 可以从 list 命令的输出里找到。

注意事项：
  - update 命令内容未变时返回 skipped，不会重复入库。
  - update 命令若中途 insert 失败，会自动回滚重插旧 chunk，不会丢数据。
  - delete 命令不可撤销，删除前请确认 doc_id 正确。
  - 运行前请确保 Docker 里的 Milvus 容器已启动（docker compose up -d）。
"""

import argparse
import hashlib
import os
import sqlite3
import sys

from documents.write_milvus import dept_from_rel_path

DB_PATH = "tracing.db"
MD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datas", "md")


def _get_registry_all() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT doc_id, filename, dept_id, chunk_count, content_hash, updated_at "
        "FROM document_registry ORDER BY filename"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_registry_one(doc_id: str) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM document_registry WHERE doc_id = ?", [doc_id]
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_milvus_chunk_count(doc_id: str, mv) -> int:
    col = mv.vector_store_saved.col
    rows = col.query(
        expr=f"doc_id == '{doc_id}'",
        output_fields=["id"],
        consistency_level="Strong",
    )
    return len(rows)


# ── 子命令：list ──────────────────────────────────────────────────────────────

def cmd_list(_args):
    rows = _get_registry_all()
    if not rows:
        print("document_registry 为空，尚未入库任何文档。")
        print("运行 python -m documents.write_milvus 进行首次批量入库。")
        return

    print(f"\n{'doc_id':<20} {'filename':<25} {'dept_id':<14} {'chunks':>6}  {'updated_at'}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['doc_id']:<20} {r['filename']:<25} {r['dept_id']:<14} "
            f"{r['chunk_count']:>6}  {r['updated_at'][:19]}"
        )
    print(f"\n共 {len(rows)} 篇文档")


# ── 子命令：status ────────────────────────────────────────────────────────────

def cmd_status(args):
    doc_id = args.doc_id
    rec = _get_registry_one(doc_id)

    if rec is None:
        print(f"doc_id={doc_id} 不在 document_registry 中。")
        print("用 python manage_docs.py list 查看所有已注册文档。")
        return

    from documents.milvus_db import MilvusVectorSave
    mv = MilvusVectorSave()
    mv.create_connection()
    milvus_count = _get_milvus_chunk_count(doc_id, mv)

    print(f"\ndoc_id      : {rec['doc_id']}")
    print(f"filename    : {rec['filename']}")
    print(f"dept_id     : {rec['dept_id']}")
    print(f"content_hash: {rec['content_hash']}")
    print(f"chunk_count : registry={rec['chunk_count']}  milvus实际={milvus_count}", end="")
    if rec['chunk_count'] != milvus_count:
        print("  ⚠ 不一致，建议重新 update")
    else:
        print()
    print(f"created_at  : {rec['created_at'][:19]}")
    print(f"updated_at  : {rec['updated_at'][:19]}")


# ── 子命令：update ────────────────────────────────────────────────────────────

def cmd_update(args):
    file_path = args.file_path
    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        sys.exit(1)

    # doc_id 必须用相对 datas/md 的路径算，和 write_milvus.py 批量入库时的算法保持一致，
    # 否则同一篇文档在这里算出的 doc_id 和批量入库时的对不上，update 会被误判成新文档
    rel_path = os.path.relpath(os.path.abspath(file_path), MD_DIR)
    doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:16]
    dept_id = args.dept or dept_from_rel_path(rel_path)

    print(f"文件     : {file_path}")
    print(f"doc_id   : {doc_id}")
    print(f"dept_id  : {dept_id}")

    from documents.milvus_db import MilvusVectorSave
    from documents.lifecycle import upsert_document
    mv = MilvusVectorSave()
    mv.create_connection()

    result = upsert_document(doc_id, file_path, dept_id, mv, source_path=rel_path)

    if result == "skipped":
        print("结果: 内容未变，跳过（无需更新）")
    elif result == "created":
        print("结果: 新增入库成功")
    elif result == "updated":
        print("结果: 更新成功（旧 chunk 已替换）")


# ── 子命令：delete ────────────────────────────────────────────────────────────

def cmd_delete(args):
    doc_id = args.doc_id
    rec = _get_registry_one(doc_id)

    if rec is None:
        print(f"doc_id={doc_id} 不在 document_registry 中，无需删除。")
        return

    print(f"即将删除: {rec['filename']}  doc_id={doc_id}  chunk 数={rec['chunk_count']}")
    confirm = input("确认删除？输入 yes 继续，其他任意键取消: ").strip().lower()
    if confirm != "yes":
        print("已取消。")
        return

    from documents.milvus_db import MilvusVectorSave
    from documents.lifecycle import delete_document
    mv = MilvusVectorSave()
    mv.create_connection()

    delete_document(doc_id, mv)
    print(f"删除完成: {rec['filename']}")


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="manage_docs",
        description="知识库文档管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出所有已注册文档")

    p_status = sub.add_parser("status", help="查看某篇文档的详细状态")
    p_status.add_argument("doc_id", help="文档 ID（从 list 命令获取）")

    p_update = sub.add_parser("update", help="新增或更新一篇文档（幂等）")
    p_update.add_argument("file_path", help="Markdown 文件路径")
    p_update.add_argument("--dept", help="部门权限标签（不传则按文件名自动映射）")

    p_delete = sub.add_parser("delete", help="删除一篇文档的所有 chunk 和注册记录")
    p_delete.add_argument("doc_id", help="文档 ID（从 list 命令获取）")

    args = parser.parse_args()
    {"list": cmd_list, "status": cmd_status, "update": cmd_update, "delete": cmd_delete}[args.cmd](args)


if __name__ == "__main__":
    main()
