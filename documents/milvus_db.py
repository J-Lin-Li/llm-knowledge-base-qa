from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import IndexType, MilvusClient, Function
from pymilvus.client.types import MetricType, DataType, FunctionType

from documents.markdown_parser import MarkdownParser
from llm_models.embeddings_model import bge_embedding
from utils.env_utils import MILVUS_URI, COLLECTION_NAME
from utils.log_utils import log

# schema 里声明过的 metadata 字段白名单（不含 id/text/dense/sparse——这几个不是从
# metadata 读的：id 是 auto_id，text 来自 page_content，dense/sparse 是算出来的）。
# langchain_milvus 写入时对不在这个集合里的 key 会静默丢弃（不报错），add_documents
# 显式过滤一遍不是修复什么 bug，是把"塞什么就丢什么"这个隐式行为变成"我决定哪些
# 字段入库"——Unstructured 以后加新 metadata 字段时会在日志里看见，不会毫无察觉。
_METADATA_FIELDS = frozenset({
    'category', 'source', 'filename', 'filetype', 'title', 'category_depth',
    'dept_id', 'doc_id', 'parent_id', 'chunk_offset', 'section_seq',
})


class MilvusVectorSave:
    """把新的document数据插入到数据库中"""

    def __init__(self) -> object:
        """自定义collection的索引"""
        self.vector_store_saved: Milvus = None

    def create_collection(self):
        client = MilvusClient(uri=MILVUS_URI)
        schema = client.create_schema()
        schema.add_field(field_name='id', datatype=DataType.INT64, is_primary=True, auto_id=True)
        # lowercase filter：修复 BM25 大小写敏感问题（实测确认 "CrashLoopBackOff" 精确大小写
        # 能匹配到正确文档，但 "crashloopbackoff" 全小写查询 0 命中——因为原来的 filter 只做
        # cnalphanumonly（去符号），不做大小写归一化。这个改动只影响新写入数据的分词结果，
        # 已入库的 t_collection01 用旧 analyzer 算好的稀疏向量不会自动重算，需要整库重新入库才生效。
        schema.add_field(field_name='text', datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True,
                         analyzer_params={"tokenizer": "jieba", "filter": ["lowercase", "cnalphanumonly"]})
        schema.add_field(field_name='category', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filename', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='filetype', datatype=DataType.VARCHAR, max_length=1000)
        schema.add_field(field_name='title', datatype=DataType.VARCHAR, max_length=1000, nullable=True)
        schema.add_field(field_name='category_depth', datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name='dept_id', datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name='doc_id', datatype=DataType.VARCHAR, max_length=64)
        # small-to-big 父子块架构（见 CLAUDE.md）：parent_id 为空表示该章节本身
        # 未被切分（<=300字），小块=章节本身，不建父块记录，不查 SQLite。
        # chunk_offset 是该小块在其父块内的顺序位置，切割时直接写入（不做事后
        # 文本匹配——小块经 SemanticChunker 切出后可能有空白/换行规范化，与父块
        # 原文不保证逐字相同，事后匹配会静默失败）。
        schema.add_field(field_name='parent_id', datatype=DataType.VARCHAR, max_length=80, nullable=True)
        schema.add_field(field_name='chunk_offset', datatype=DataType.INT64, nullable=True)
        # section_seq：小块所属章节在文档内的原始顺序号，不管章节有没有被切分/建父块
        # 都有值——D7 排序（同文档内按章节原序）对所有 context 项统一适用。
        schema.add_field(field_name='section_seq', datatype=DataType.INT64, nullable=True)
        schema.add_field(field_name='sparse', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='dense', datatype=DataType.FLOAT_VECTOR, dim=512)

        bm25_function = Function(
            name="text_bm25_emb",  # Function name
            input_field_names=["text"],  # Name of the VARCHAR field containing raw text data
            output_field_names=["sparse"],
            # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
            function_type=FunctionType.BM25,  # Set to `BM25`
        )
        schema.add_function(bm25_function)
        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="sparse",
            index_name="sparse_inverted_index",#倒排索引
            index_type="SPARSE_INVERTED_INDEX",  # Inverted index type for sparse vectors，倒排索引
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                # Algorithm for building and querying the index. Valid values: DAAT_MAXSCORE, DAAT_WAND, TAAT_NAIVE.
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            },
        )
        index_params.add_index(
            field_name="dense",
            index_name="dense_inverted_index",
            index_type=IndexType.HNSW,  # Inverted index type for sparse vectors
            metric_type=MetricType.IP,
            params={"M": 16, "efConstruction": 64}  # M :邻接节点数, efConstruction: 搜索范围
        )

        if COLLECTION_NAME in client.list_collections():
            # 先释放， 再删除索引，再删除collection
            client.release_collection(collection_name=COLLECTION_NAME)
            client.drop_index(collection_name=COLLECTION_NAME, index_name='sparse_inverted_index')
            client.drop_index(collection_name=COLLECTION_NAME, index_name='dense_inverted_index')
            client.drop_collection(collection_name=COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def create_connection(self):
        """创建一个Connection： milvus + langchain。pip install  langchain-milvus"""
        self.vector_store_saved = Milvus(
            embedding_function=bge_embedding,
            collection_name=COLLECTION_NAME,
            builtin_function=BM25BuiltInFunction(),
            vector_field=['dense', 'sparse'],
            consistency_level="Strong",
            auto_id=True,
            connection_args={"uri": MILVUS_URI}
        )

    def add_documents(self, datas: List[Document]):
        """把新的document保存到Milvus中，写入前显式过滤 metadata 到 schema 白名单。"""
        dropped_keys = set()
        for doc in datas:
            extra = set(doc.metadata.keys()) - _METADATA_FIELDS
            if extra:
                dropped_keys |= extra
                doc.metadata = {k: v for k, v in doc.metadata.items() if k in _METADATA_FIELDS}
        if dropped_keys:
            log.warning(f"add_documents: 以下 metadata 字段不在 schema 白名单内，已过滤: {sorted(dropped_keys)}")
        self.vector_store_saved.add_documents(datas)

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Document]:
        """查询某篇文档的所有 chunk，用于删旧前保存快照以支持回滚。"""
        _FIELDS = ["text", "category", "source", "filename", "filetype",
                   "title", "category_depth", "dept_id", "doc_id",
                   "parent_id", "chunk_offset", "section_seq"]
        col = self.vector_store_saved.col
        rows = col.query(
            expr=f"doc_id == '{doc_id}'",
            output_fields=_FIELDS,
            consistency_level="Strong",
        )
        return [
            Document(
                page_content=r["text"],
                metadata={k: r[k] for k in _FIELDS if k != "text" and r.get(k) is not None},
            )
            for r in rows
        ]

    def delete_by_doc_id(self, doc_id: str) -> int:
        """删除某篇文档的所有 chunk，返回删除数量。"""
        col = self.vector_store_saved.col
        rows = col.query(
            expr=f"doc_id == '{doc_id}'",
            output_fields=["id"],
            consistency_level="Strong",
        )
        if not rows:
            return 0
        ids = [r["id"] for r in rows]
        col.delete(expr=f"id in {ids}")
        return len(ids)



if __name__ == '__main__':
    # 解析文件内容
    file_path = r'E:\my_project\RAG_PROJECT\datas\md\tech_report_0tfhhamx.md'
    parser = MarkdownParser()
    docs = parser.parse_markdown_to_documents(file_path)

    # 写入Milvus数据库
    mv = MilvusVectorSave()
    mv.create_collection()
    mv.create_connection()
    mv.add_documents(docs)

    client = mv.vector_store_saved.client
    # 得到表结构
    desc_collection = client.describe_collection(
        collection_name=COLLECTION_NAME
    )
    print('表结构是: ', desc_collection)

    # 得到当前表的，所有的index
    res = client.list_indexes(
        collection_name=COLLECTION_NAME
    )
    print('表中的所有索引：', res)

    if res:
        for i in res:
            # 得到索引的描述
            desc_index = client.describe_index(
                collection_name=COLLECTION_NAME,
                index_name=i
            )
            print(desc_index)

    result = client.query(
        collection_name=COLLECTION_NAME,
        filter="category == 'Title'",  # 查询 category == 'Title' 的所有数据
        output_fields=['text', 'category', 'filename']  # 指定返回的字段
    )

    print('测试 过滤查询的结果是: ', result)
