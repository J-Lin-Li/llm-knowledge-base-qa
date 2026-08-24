from langchain_core.runnables import RunnableConfig

from graph2.grade_answer_chain import answer_grader_chain
from utils.log_utils import log


def check_answer(state, config: RunnableConfig = None):
    """
    答案质量评估节点。

    config["configurable"]["enable_answer_check"] 为 False 时跳过 LLM 调用，
    直接写 "yes"（视为已解决），后台质量监控由 FastAPI BackgroundTasks 异步处理。
    默认值 True（CLI 模式保持完整闭环）。
    """
    configurable = (config or {}).get("configurable", {})
    if not configurable.get("enable_answer_check", True):
        log.info("---答案质量评估：已禁用，默认通过（后台异步监控）---")
        return {"answer_result": "yes"}

    log.info("---答案质量评估---")
    question = state["question"]
    generation = state["generation"]

    score = answer_grader_chain.invoke({"question": question, "generation": generation})
    result = score.binary_score
    log.info(f"---答案质量评估结果: {'解决' if result == 'yes' else '未解决'}---")
    return {"answer_result": result}
