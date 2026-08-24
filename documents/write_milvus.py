import hashlib
import multiprocessing
import os
from multiprocessing import Queue

from datetime import datetime, timezone

from documents.markdown_parser import MarkdownParser, assign_parent_ids
from documents.milvus_db import MilvusVectorSave
from tracing.db import upsert_parent_chunk
from utils.log_utils import log
import queue

# 目录前缀 → 部门权限映射（演示用，生产环境应从数据库读取）
# K8s 文档命名大量重复（每个子目录都有 index.md），按文件名映射已失效，改为按
# 相对 datas/md 的一级子目录（如 concepts/security、tasks/tls）分类
DIR_TO_DEPT = {
    # ops：集群运维、调度、网络与存储管理
    "concepts/cluster-administration": "ops",
    "concepts/scheduling-eviction": "ops",
    "concepts/storage": "ops",
    "concepts/services-networking": "ops",
    "tasks/administer-cluster": "ops",
    "tasks/debug": "ops",
    "tasks/manage-daemon": "ops",
    "tasks/manage-gpus": "ops",
    "tasks/manage-hugepages": "ops",
    "tasks/network": "ops",
    # engineering：安全、配置、扩展开发
    "concepts/security": "engineering",
    "concepts/policy": "engineering",
    "concepts/configuration": "engineering",
    "concepts/extend-kubernetes": "engineering",
    "concepts/windows": "engineering",
    "tasks/configmap-secret": "engineering",
    "tasks/configure-pod-container": "engineering",
    "tasks/extend-kubernetes": "engineering",
    "tasks/extend-kubectl": "engineering",
    "tasks/inject-data-application": "engineering",
    "tasks/tls": "engineering",
    # product：工作负载、应用访问
    "concepts/workloads": "product",
    "tasks/access-application-cluster": "product",
    "tasks/run-application": "product",
    "tasks/job": "product",
    # public：基础概念、通用工具
    "concepts/architecture": "public",
    "concepts/overview": "public",
    "concepts/containers": "public",
    "tasks/manage-kubernetes-objects": "public",
    "tasks/tools": "public",
}


def dept_from_rel_path(rel_path: str) -> str:
    """按 rel_path 的一级子目录（如 concepts/security）查部门映射，查不到默认 public"""
    parts = rel_path.replace(os.sep, "/").split("/")
    if len(parts) < 2:
        return "public"
    return DIR_TO_DEPT.get(f"{parts[0]}/{parts[1]}", "public")


# 采用分布式，多进程的方式把海量数据写入Milvus数据库

def file_parser_process(dir_path: str, output_queue: Queue, batch_size: int = 20):
    """进程1：递归扫描目录下所有md文件并分批放入队列"""
    log.info(f"解析进程开始扫描目录: {dir_path}")

    # 递归获取所有子目录下的 .md 文件
    md_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.md'):
                md_files.append(os.path.join(root, f))

    if not md_files:
        log.warning("警告：未找到任何.md文件")
        output_queue.put(None)  # 发送终止信号
        return

    parser = MarkdownParser()
    doc_batch = []
    parent_batch_count = 0
    for file_path in md_files:
        try:
            chunks, parent_records = parser.parse_markdown_to_documents(file_path)
            if chunks:
                # 用相对路径生成 doc_id，避免不同子目录下同名文件（如 index.md）产生冲突
                rel_path = os.path.relpath(file_path, dir_path)
                dept_id = dept_from_rel_path(rel_path)
                doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:16]
                for doc in chunks:
                    doc.metadata["dept_id"] = dept_id
                    doc.metadata["doc_id"] = doc_id
                    # source 覆盖成相对路径：doc_id 是哈希不可读，K8s 大量重名 index.md
                    # 单靠 filename 分不清，相对路径可以直接看出是哪个文件，且不受
                    # 运行机器的绝对路径前缀影响
                    doc.metadata["source"] = rel_path

                chunks, parent_records = assign_parent_ids(chunks, parent_records, doc_id)

                # 父块直接在本进程写 SQLite（本地文件，parser 进程内部顺序写，
                # 和 writer 进程各自独立连接，不会有并发写冲突——写的是不同的库）
                now = datetime.now(timezone.utc).isoformat()
                for record in parent_records:
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
                parent_batch_count += len(parent_records)

                doc_batch.extend(chunks)

            # 达到批次大小时发送 到队列中
            if len(doc_batch) >= batch_size:
                output_queue.put(doc_batch.copy())
                doc_batch.clear()  # 清空当前缓冲区的所有批次数据
        except Exception as e:
            log.error(f"解析失败 {file_path}: {str(e)}")
            log.exception(e)

    # 发送剩余文档
    if doc_batch:
        output_queue.put(doc_batch)

    # 发送终止信号
    output_queue.put(None)
    log.info(f"解析完成，共处理{len(md_files)}个文件，产生父块{parent_batch_count}个")


def milvus_writer_process(input_queue: Queue):
    """进程2：从队列读取并写入Milvus"""
    log.info("Milvus写入进程启动...")


    mv = MilvusVectorSave()
    mv.create_connection()
    total_count = 0
    while True:
        try:

            # datas = input_queue.get() #若是进程1挂掉了，会无限等待，改为如下：
            try:
                datas = input_queue.get(timeout=30)
            except queue.Empty:
                log.error("队列等待超时，解析进程可能已异常退出")
                break


            if datas is None:  # 收到了终止的信号
                break

            if isinstance(datas, list):
                mv.add_documents(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")
        except Exception as e:
            log.error(f"写入数据是吧 ！")
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


def _register_all_docs(md_dir: str, mv: MilvusVectorSave) -> None:
    """写入完成后，把每个 md 文件的 doc_id / content_hash 注册到 document_registry。"""
    from datetime import datetime, timezone
    from tracing.db import upsert_doc_record

    # 递归收集所有 md 文件的相对路径
    md_files = []
    for root, _, files in os.walk(md_dir):
        for f in files:
            if f.endswith('.md'):
                md_files.append(os.path.relpath(os.path.join(root, f), md_dir))

    col = mv.vector_store_saved.col
    now = datetime.now(timezone.utc).isoformat()

    for rel_path in md_files:
        file_path = os.path.join(md_dir, rel_path)
        fname = os.path.basename(rel_path)
        with open(file_path, encoding='utf-8') as f:
            content = f.read()
        content_hash = hashlib.md5(content.encode()).hexdigest()
        doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:16]
        dept_id = dept_from_rel_path(rel_path)

        rows = col.query(
            expr=f"doc_id == '{doc_id}'",
            output_fields=["id"],
            consistency_level="Strong",
        )
        upsert_doc_record({
            "doc_id": doc_id,
            "filename": fname,
            "dept_id": dept_id,
            "content_hash": content_hash,
            "chunk_count": len(rows),
            "created_at": now,
            "updated_at": now,
        })
    log.info(f"document_registry 注册完成，共 {len(md_files)} 个文档")


if __name__ == '__main__':
    # 配置参数
    md_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datas', 'md')
    queue_maxsize = 20  # 队列最大容量（防止内存溢出）

    mv = MilvusVectorSave()
    mv.create_collection()

    # 创建进程间通信队列
    docs_queue = Queue(maxsize=queue_maxsize)

    # 启动子进程
    parser_proc = multiprocessing.Process(
        target=file_parser_process,
        args=(md_dir, docs_queue)
    )
    writer_proc = multiprocessing.Process(
        target=milvus_writer_process,
        args=(docs_queue,)
    )

    parser_proc.start()
    writer_proc.start()

    # 等待进程结束
    parser_proc.join()
    writer_proc.join()

    # 写入完成后注册 document_registry
    mv.create_connection()
    _register_all_docs(md_dir, mv)

    print("系统提示：所有任务完成")