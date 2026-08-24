from langchain_core.runnables import RunnableConfig

from graph2.grade_hallucinations_chain import hallucination_grader_chain
from utils.log_utils import log


def check_hallucination(state, config: RunnableConfig = None):
    """
    幻觉检测节点。

    config["configurable"]["enable_hallucination_check"] 为 False 时跳过 LLM 调用，
    直接写 "yes"（视为无幻觉），后台质量监控由 FastAPI BackgroundTasks 异步处理。
    默认值 True（CLI 模式保持完整闭环）。
    """
    configurable = (config or {}).get("configurable", {})
    if not configurable.get("enable_hallucination_check", True):
        log.info("---幻觉检测：已禁用，默认通过（后台异步监控）---")
        return {"hallucination_result": "yes"}

    log.info("---幻觉检测---")
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
    result = score.binary_score
    log.info(f"---幻觉检测结果: {'无幻觉' if result == 'yes' else '有幻觉'}---")
    return {"hallucination_result": result}
