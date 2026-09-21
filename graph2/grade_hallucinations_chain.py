from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field, BaseModel

from llm_models.all_llm import llm


# 数据模型 - 生成内容幻觉评分
class GradeHallucinations(BaseModel):
    """对生成回答中是否存在幻觉进行二元评分"""

    # 字段类型从 str 收紧为 Literal["yes","no"]（2026-09-12）：原来靠 prompt
    # 指令约束模型只填 yes/no，字段本身不是强枚举，模型往这个字段里塞解释文字
    # 在类型层面是合法的——万一真的塞了，还可能带出原文片段。改成 Literal 后，
    # 模型返回非 yes/no 会在 Pydantic 校验阶段直接报错，而不是被无声地存下来。
    # 报错本身怎么处理见 api/app.py 的 _bg_quality_check（异常信息不直接落库）。
    binary_score: Literal["yes", "no"] = Field(
        description="回答是否基于事实，取值为'yes'或'no'"
    )


# 带函数调用的LLM初始化
structured_llm_grader = llm.with_structured_output(GradeHallucinations, method="function_calling")

# 提示词模板
system = """您是一个评估生成内容是否基于检索事实的评分器。\n
     给出'yes'或'no'的二元评分。'yes'表示回答是基于/支持于给定事实集的。"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 系统角色设定
        ("human", "事实集: \n\n {documents} \n\n 生成内容: {generation}"),  # 用户输入模板
    ]
)

# 构建幻觉检测工作流
hallucination_grader_chain = (
        hallucination_prompt  # 使用幻觉检测提示模板
        | structured_llm_grader  # 调用结构化评分的LLM
)
