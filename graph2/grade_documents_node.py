from langchain_core.runnables import RunnableConfig

from utils.log_utils import log

# baseline：DeepSeek LLM 打分版本（消融实验对照组，切换时取消注释）
# from graph2.grader_chain import retrieval_grader_chain
#
# def grade_documents(state):
#     log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     documents = state["documents"]
#
#     filtered_docs = []
#     for d in documents:
#         score = retrieval_grader_chain.invoke(
#             {"question": question, "document": d.page_content}
#         )
#         grade = score.binary_score
#         if grade == "yes":
#             log.info("---GRADE: 打印相关标识---")
#             filtered_docs.append(d)
#         else:
#             log.info("---GRADE: 打印不相关标识,并丢掉doc---")
#             continue
#     return {"documents": filtered_docs, "question": question}


RERANKER_THRESHOLD = 0.3


def filter_by_reranker_score(documents, threshold):
    """按 doc.metadata['reranker_score'] 做阈值过滤，score 缺失时默认保留。

    知识库路径（grade_documents，本文件）和 web 路径（grade_web_results_node.py）
    共用这同一段逻辑，唯一的差异是传入的 threshold 常量不同（RERANKER_THRESHOLD
    vs WEB_RERANKER_THRESHOLD）——两个节点仍然分开注册、分开写 trace span，只是
    阈值比较这一步不重复写两遍。
    """
    filtered_docs = []
    for d in documents:
        score = d.metadata.get("reranker_score")
        if score is None:
            log.info("---GRADE: reranker_score 缺失，默认保留---")
            filtered_docs.append(d)
        elif score > threshold:
            log.info(f"---GRADE: 相关（score={score:.4f}）---")
            filtered_docs.append(d)
        else:
            log.info(f"---GRADE: 不相关（score={score:.4f}），丢弃---")
    return filtered_docs


def grade_documents(state, config: RunnableConfig = None):
    log.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # 阈值只比较这一次。filtered_docs 空 = 没有任何候选过阈值 = 知识库没覆盖，
    # 不管下面消融开关开不开，都是这个结果——decide_to_generate 靠"documents
    # 是不是空"触发改写/降级web，这个判断只能用这一次比较的结果，不能重复算。
    filtered_docs = filter_by_reranker_score(documents, RERANKER_THRESHOLD)

    if not filtered_docs:
        return {"documents": [], "question": question}

    configurable = (config or {}).get("configurable", {})
    enable_relevance_filter = configurable.get("enable_relevance_filter", True)

    if not enable_relevance_filter:
        # 消融：覆盖度已经由上面这次比较确认足够（filtered_docs 非空），这里不
        # 再逐条丢弃，把原始候选（未经筛选）整批带入生成——测的是"逐条过滤"这
        # 一步本身的价值，和"要不要放弃知识库"互不影响。
        log.info(f"---相关性过滤已禁用（消融），保留全部 {len(documents)} 个候选---")
        return {"documents": documents, "question": question}

    return {"documents": filtered_docs, "question": question}
