import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "tracing.db"

_DDL = """
CREATE TABLE IF NOT EXISTS spans (
    trace_id          TEXT NOT NULL,
    span_id           TEXT NOT NULL PRIMARY KEY,
    parent_span_id    TEXT,
    thread_id         TEXT,
    node_name         TEXT,
    span_type         TEXT,
    model_name        TEXT,
    input             TEXT,
    output            TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    status            TEXT DEFAULT 'ok',
    error_msg         TEXT,
    extra             TEXT,
    created_at        TEXT NOT NULL,
    execution_index   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_spans_trace  ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_node   ON spans(node_name);
CREATE INDEX IF NOT EXISTS idx_spans_thread ON spans(thread_id);

CREATE TABLE IF NOT EXISTS document_registry (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    dept_id      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_registry_filename ON document_registry(filename);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    first_question TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

-- small-to-big 父子块架构：父块内容存这里，不进 Milvus metadata（父块不参与
-- 任何向量计算，是按 key 查询的文本存储，进 Milvus 会导致同一份父块文本被
-- 它的每个子块重复存储）。只有真正被切分过（产出>=2个小块）的章节才有记录，
-- "切不动"的章节不建父块（见 CLAUDE.md）。
CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id    TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    dept_id      TEXT NOT NULL,
    title        TEXT,
    content      TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    seq_in_doc   INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_parent_chunks_doc ON parent_chunks(doc_id);
"""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_DDL)


def insert_span(span: dict) -> None:
    cols = ", ".join(span.keys())
    placeholders = ", ".join(f":{k}" for k in span.keys())
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(f"INSERT INTO spans ({cols}) VALUES ({placeholders})", span)


def update_span(span_id: str, **fields) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"UPDATE spans SET {set_clause} WHERE span_id = ?",
            [*fields.values(), span_id],
        )


# ── document_registry CRUD ────────────────────────────────────────────────────

def get_doc_record(doc_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM document_registry WHERE doc_id = ?", [doc_id]
        ).fetchone()
    return dict(row) if row else None


def upsert_doc_record(record: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO document_registry
                (doc_id, filename, dept_id, content_hash, chunk_count, created_at, updated_at)
            VALUES
                (:doc_id, :filename, :dept_id, :content_hash, :chunk_count, :created_at, :updated_at)
            ON CONFLICT(doc_id) DO UPDATE SET
                filename     = excluded.filename,
                dept_id      = excluded.dept_id,
                content_hash = excluded.content_hash,
                chunk_count  = excluded.chunk_count,
                updated_at   = excluded.updated_at
            """,
            record,
        )


def delete_doc_record(doc_id: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM document_registry WHERE doc_id = ?", [doc_id])


# ── parent_chunks CRUD ────────────────────────────────────────────────────────

def upsert_parent_chunk(record: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO parent_chunks
                (parent_id, doc_id, dept_id, title, content, char_count, seq_in_doc, created_at, updated_at)
            VALUES
                (:parent_id, :doc_id, :dept_id, :title, :content, :char_count, :seq_in_doc, :created_at, :updated_at)
            ON CONFLICT(parent_id) DO UPDATE SET
                doc_id     = excluded.doc_id,
                dept_id    = excluded.dept_id,
                title      = excluded.title,
                content    = excluded.content,
                char_count = excluded.char_count,
                seq_in_doc = excluded.seq_in_doc,
                updated_at = excluded.updated_at
            """,
            record,
        )


def get_parent_chunk(parent_id: str) -> dict | None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM parent_chunks WHERE parent_id = ?", [parent_id]
        ).fetchone()
    return dict(row) if row else None


def get_parent_chunks_by_doc(doc_id: str) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM parent_chunks WHERE doc_id = ?", [doc_id]
        ).fetchall()
    return [dict(r) for r in rows]


def delete_parent_chunks_by_doc(doc_id: str) -> int:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM parent_chunks WHERE doc_id = ?", [doc_id])
        return cur.rowcount


# ── sessions CRUD ─────────────────────────────────────────────────────────────

def create_session(session_id: str, first_question: str) -> None:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, first_question, created_at) VALUES (?, ?, ?)",
            [session_id, first_question, now],
        )


def list_sessions(limit: int = 10) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT session_id, first_question, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
    return [dict(r) for r in rows]


init_db()
