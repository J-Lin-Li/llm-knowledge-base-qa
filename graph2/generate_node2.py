from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm_models.all_llm import llm

_BASE_TEMPLATE = (
    "你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。"
    "如果不知道答案，请直接说明。回答保持简洁。\n"
    "问题：{question} \n上下文：{context} \n回答："
)

# 幻觉重试专用模板：只在 check_hallucination 判过一次"有幻觉"、触发重新生成时
# 才用这版——第一次生成（generation_count=0）不用，避免所有问题的首次回答都
# 变得过度保守。原来的重试是拿完全相同的 prompt 再调一次 LLM，temperature=0
# 下模型倾向不变，重试基本救不回来（消融实验验证过：faithfulness 开关门禁前后
# 完全持平，0.7188=0.7188）。这版重试显式要求"只说上下文里明确写的"，尝试
# 真正改变输入而不是原样重复。
_GROUNDED_RETRY_TEMPLATE = (
    "你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。"
    "注意：只能陈述上下文中明确提到的信息；如果上下文没有涵盖问题所需的某个细节，"
    "直接说明该信息在提供的资料中未提及，不要用你自己的知识去推测或补充。"
    "回答保持简洁。\n问题：{question} \n上下文：{context} \n回答："
)


def format_docs(documents, parent_contexts):
    """
    格式化喂给 LLM 的上下文——small-to-big：优先用 parent_contexts（父块内容，已按
    D7 排序、去重）；web_search 路径没有 parent_contexts（不经过 assemble_context），
    回退到小块本身——web_search 节点现在也总是返回 Document 列表（精排后的若干段），
    和知识库路径同构，不再有"单个 Document 不是列表"的情况。

    模块级函数（2026-09-12 从 generate() 内部提出来，供外部复用）：generate 真正读的
    就是这个函数的输出。任何需要还原"LLM 实际看到了什么"的地方（比如异步质检要判断
    答案有没有依据）都必须调用这同一个函数，不能自己另写一份格式化逻辑——否则会重蹈
    api/app.py 的 _bg_quality_check 曾经踩过的坑：判断依据（小块）和 generate 真正用的
    材料（父块）不是同一份，判断结果没有意义。
    """
    if parent_contexts:
        return "\n\n".join(item["content"] for item in parent_contexts)
    return "\n\n".join(doc.page_content for doc in documents)


def generate(state):
    """
    生成回答
    Args:
        state (dict): 当前图状态，包含问题和检索结果
    Returns:
        state (dict): 更新后的状态，新增包含生成结果的generation字段
    """
    question = state["question"]  # 获取用户问题
    documents = state["documents"]  # 检索到的小块（web_search 路径下用它兜底）
    parent_contexts = state.get("parent_contexts")  # assemble_context 写入的父块级 context
    generation_count = state.get("generation_count", 0)

    template = _GROUNDED_RETRY_TEMPLATE if generation_count > 0 else _BASE_TEMPLATE
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    # 构建RAG处理链
    rag_chain = (
            prompt |  # 第一步：使用提示模板
            llm |  # 第二步：调用语言模型
            StrOutputParser()  # 第三步：解析模型输出为字符串
    )

    # RAG生成过程
    context = format_docs(documents, parent_contexts)
    generation = rag_chain.invoke({"context": context, "question": question})  # 调用RAG链生成回答
    return {"documents": documents, "question": question, "generation": generation, "generation_count": generation_count + 1}