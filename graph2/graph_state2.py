from typing import TypedDict, List, Optional

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    表示图处理流程的状态信息

    属性说明：
        question: 当前用于检索/生成的问题文本（会被 transform_query 覆写为改写后的版本）
        original_question: 用户最初提出的问题文本，全程不变，供 transform_query 每次都从原始意图重写
        generation: 语言模型生成的回答文本
        transform_count: 传换查询的次数
        documents: 检索到的相关文档列表（small-to-big架构下是"小块"，检索单元）
        parent_contexts: assemble_context 节点写入，供 generate 读取——去重、排序后的
            父块（或"切不动"章节自身的小块）内容列表，是真正喂给 LLM 的 context
        hallucination_result: 幻觉检测节点写入的判断结果（"yes"=无幻觉/"no"=有幻觉）
        answer_result: 答案质量节点写入的判断结果（"yes"=解决/"no"=未解决）
    """

    question: str  # 存储当前处理的用户问题（可能已被改写）
    original_question: str  # 存储最初的用户问题，不随改写变化
    transform_count: int  # 无文档路的改写次数
    not_useful_count: int  # 答案质量失败路的改写次数
    generation_count: int  # 幻觉重试次数
    generation: str  # 存储LLM生成的回答内容
    documents: List[Document]  # 存储检索到的小块列表（检索单元）
    parent_contexts: Optional[List[dict]]  # assemble_context 写入的父块级 context
    hallucination_result: str  # check_hallucination 节点写入，供条件边读取
    answer_result: str  # check_answer 节点写入，供条件边读取