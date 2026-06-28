from langchain_core.tools import create_retriever_tool
from documents.milvus_db import MilvusVectorSave

mv = MilvusVectorSave()
mv.create_connection()
retriever = mv.vector_store_saved.as_retriever(
    search_type='similarity',  # 仅返回相似度超过阈值的文档
    search_kwargs={
        "k": 4, #返回文档数
        "score_threshold": 0.1,             #得分阈值，低于则丢弃
        "ranker_type": "rrf",               #Reciprocal Rank Fusion，倒数排名融合，例：排名1，3，则分数=1/101+1/103
        "ranker_params": {"k": 100},        #倒数排名融合底数基数
        # "param": {"ef": 20},#候选池
        'filter': {"category": "content"}   #过滤，只检索content，而非Title
    }
)


retriever_tool = create_retriever_tool(
    retriever,
    'rag_retriever',
    ‘搜索并返回关于 Milvus 向量数据库的信息，内容涵盖：基本概念、架构介绍、部署、配置、性能调优、FAQ 和故障排查’
)