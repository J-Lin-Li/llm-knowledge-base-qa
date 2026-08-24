"""
测试 graph_2.py 里的三个决策函数。

重构说明：
  原 grade_generation_v_documents_and_question 已拆分为：
    - check_hallucination 节点（调用 LLM）→ 写 state["hallucination_result"]
    - check_answer 节点（调用 LLM）→ 写 state["answer_result"]
    - route_from_hallucination 路由函数（纯逻辑，读 state）
    - route_from_answer 路由函数（纯逻辑，读 state）

  route_from_* 是纯函数，只读 state dict，无需 mock。
  check_* 节点调用 LLM chain，需要 @patch 拦截。

被测函数：
  - decide_to_generate       ：文档打分后路由
  - route_from_hallucination ：幻觉检测后路由（generation_count 检查）
  - route_from_answer        ：答案质量后路由（not_useful_count 检查）
  - check_hallucination      ：节点函数，调用 hallucination_grader_chain
  - check_answer             ：节点函数，调用 answer_grader_chain
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from graph2.graph_2 import (
    decide_to_generate,
    route_from_hallucination,
    route_from_answer,
    route_after_web_grade,
    web_search_fallback,
    hallucination_fallback,
)
from graph2.check_hallucination_node import check_hallucination
from graph2.check_answer_node import check_answer


# ═══════════════════════════════════════════════════════
# decide_to_generate — 纯逻辑，无需 mock
# ═══════════════════════════════════════════════════════

class TestDecideToGenerate:

    def test_有相关文档_走生成节点(self):
        state = {"documents": [Document(page_content="Milvus部署文档内容")], "transform_count": 0}
        assert decide_to_generate(state) == "generate"

    def test_无文档且改写未到上限_走查询改写(self):
        state = {"documents": [], "transform_count": 0}
        assert decide_to_generate(state) == "transform_query"

    def test_无文档且改写刚好到上限_降级网络搜索(self):
        state = {"documents": [], "transform_count": 1}
        assert decide_to_generate(state) == "web_search"

    def test_无文档且改写超过上限_同样降级网络搜索(self):
        state = {"documents": [], "transform_count": 5}
        assert decide_to_generate(state) == "web_search"

    def test_state没有transform_count字段_视为0(self):
        state = {"documents": []}
        assert decide_to_generate(state) == "transform_query"


# ═══════════════════════════════════════════════════════
# route_from_hallucination — 纯逻辑，无需 mock
# generation_count 由 generate 节点自增后写入 state，此处只读
# ═══════════════════════════════════════════════════════

class TestRouteFromHallucination:

    def test_无幻觉_进入答案质量评估(self):
        state = {"hallucination_result": "yes", "generation_count": 1}
        assert route_from_hallucination(state) == "check_answer"

    def test_有幻觉且未到上限_重新生成(self):
        # generation_count=2（第2次生成后），还未到3，应重试
        state = {"hallucination_result": "no", "generation_count": 2}
        assert route_from_hallucination(state) == "generate"

    def test_有幻觉且刚好到上限_触发兜底(self):
        # generation_count=3（第3次生成后），触发 fallback
        state = {"hallucination_result": "no", "generation_count": 3}
        assert route_from_hallucination(state) == "hallucination_fallback"

    def test_有幻觉且超过上限_同样触发兜底(self):
        state = {"hallucination_result": "no", "generation_count": 99}
        assert route_from_hallucination(state) == "hallucination_fallback"

    def test_幻觉兜底在第3次而非第2次(self):
        # generation_count=2：还能重试
        assert route_from_hallucination(
            {"hallucination_result": "no", "generation_count": 2}
        ) == "generate"
        # generation_count=3：触发兜底（不是 4）
        assert route_from_hallucination(
            {"hallucination_result": "no", "generation_count": 3}
        ) == "hallucination_fallback"

    def test_缺hallucination_result字段_默认no有幻觉(self):
        # 缺字段时 state.get 返回 "no"（有幻觉），generation_count=1 < 3 → 重试
        state = {"generation_count": 1}
        assert route_from_hallucination(state) == "generate"


# ═══════════════════════════════════════════════════════
# route_from_answer — 纯逻辑，无需 mock
# not_useful_count 由 transform_query 自增，此处读到的是自增前的值
# ═══════════════════════════════════════════════════════

class TestRouteFromAnswer:

    def test_答案解决了问题_返回useful(self):
        state = {"answer_result": "yes", "not_useful_count": 0}
        assert route_from_answer(state) == "useful"

    def test_答案未解决且未到上限_走查询改写(self):
        # not_useful_count=1，还未到2，transform_query 会把它变成2
        state = {"answer_result": "no", "not_useful_count": 1}
        assert route_from_answer(state) == "not useful"

    def test_答案未解决且刚好到上限_触发兜底(self):
        # not_useful_count=2，本轮不再改写，直接兜底
        state = {"answer_result": "no", "not_useful_count": 2}
        assert route_from_answer(state) == "transform_max_retries"

    def test_答案未解决且超过上限_同样触发兜底(self):
        state = {"answer_result": "no", "not_useful_count": 99}
        assert route_from_answer(state) == "transform_max_retries"

    def test_答案质量兜底在第2次改写后而非第1次(self):
        # not_useful_count=1：还能改写（改写后变2）
        assert route_from_answer(
            {"answer_result": "no", "not_useful_count": 1}
        ) == "not useful"
        # not_useful_count=2：触发兜底（不是3）
        assert route_from_answer(
            {"answer_result": "no", "not_useful_count": 2}
        ) == "transform_max_retries"

    def test_缺answer_result字段_默认no未解决(self):
        state = {"not_useful_count": 0}
        assert route_from_answer(state) == "not useful"


# ═══════════════════════════════════════════════════════
# check_hallucination 节点 — 调用 LLM，需要 @patch
# patch 目标：使用处（check_hallucination_node），不是定义处
# ═══════════════════════════════════════════════════════

class TestCheckHallucinationNode:

    def _state(self):
        return {
            "documents": [Document(page_content="Milvus 部署需要 Docker Compose。")],
            "generation": "使用 Docker Compose 部署 Milvus。",
        }

    @patch('graph2.check_hallucination_node.hallucination_grader_chain')
    def test_无幻觉_写入yes(self, mock_chain):
        mock_chain.invoke.return_value = MagicMock(binary_score="yes")
        result = check_hallucination(self._state())
        assert result == {"hallucination_result": "yes"}

    @patch('graph2.check_hallucination_node.hallucination_grader_chain')
    def test_有幻觉_写入no(self, mock_chain):
        mock_chain.invoke.return_value = MagicMock(binary_score="no")
        result = check_hallucination(self._state())
        assert result == {"hallucination_result": "no"}

    @patch('graph2.check_hallucination_node.hallucination_grader_chain')
    def test_节点只返回hallucination_result_不修改其他字段(self, mock_chain):
        mock_chain.invoke.return_value = MagicMock(binary_score="yes")
        result = check_hallucination(self._state())
        assert list(result.keys()) == ["hallucination_result"]


# ═══════════════════════════════════════════════════════
# check_answer 节点 — 调用 LLM，需要 @patch
# ═══════════════════════════════════════════════════════

class TestCheckAnswerNode:

    def _state(self):
        return {
            "question": "Milvus 如何部署？",
            "generation": "使用 Docker Compose 部署 Milvus。",
        }

    @patch('graph2.check_answer_node.answer_grader_chain')
    def test_解决了问题_写入yes(self, mock_chain):
        mock_chain.invoke.return_value = MagicMock(binary_score="yes")
        result = check_answer(self._state())
        assert result == {"answer_result": "yes"}

    @patch('graph2.check_answer_node.answer_grader_chain')
    def test_未解决问题_写入no(self, mock_chain):
        mock_chain.invoke.return_value = MagicMock(binary_score="no")
        result = check_answer(self._state())
        assert result == {"answer_result": "no"}

    @patch('graph2.check_answer_node.answer_grader_chain')
    def test_节点只返回answer_result_不修改其他字段(self, mock_chain):
        mock_chain.invoke.return_value = MagicMock(binary_score="yes")
        result = check_answer(self._state())
        assert list(result.keys()) == ["answer_result"]


# ═══════════════════════════════════════════════════════
# route_after_web_grade / web_search_fallback — web 路径的过滤后路由与降级出口
# ═══════════════════════════════════════════════════════

class TestRouteAfterWebGrade:

    def test_有文档过阈值_走生成(self):
        state = {"documents": [Document(page_content="网页内容")]}
        assert route_after_web_grade(state) == "generate"

    def test_全部被过滤_走web降级(self):
        state = {"documents": []}
        assert route_after_web_grade(state) == "web_search_fallback"

    def test_state没有documents字段_视为空_走web降级(self):
        state = {}
        assert route_after_web_grade(state) == "web_search_fallback"


class TestWebSearchFallback:

    def test_返回固定提示文案(self):
        result = web_search_fallback({})
        assert "generation" in result
        assert "未找到" in result["generation"] or "未找到与该问题相关" in result["generation"]


# ═══════════════════════════════════════════════════════
# hallucination_fallback — 幻觉检测三次未过后的降级：输出检索原文，不输出模型答案
# ═══════════════════════════════════════════════════════

class TestHallucinationFallback:

    def test_有parent_contexts_优先用父块内容并按doc_id回查source(self):
        state = {
            "documents": [
                Document(page_content="小块内容A", metadata={"doc_id": "doc-1", "source": "concepts/pod.md"}),
            ],
            "parent_contexts": [
                {"doc_id": "doc-1", "content": "父块完整内容A"},
            ],
        }
        result = hallucination_fallback(state)
        generation = result["generation"]

        assert "父块完整内容A" in generation
        assert "concepts/pod.md" in generation
        assert "小块内容A" not in generation  # 不该混入小块原文，只用父块
        assert "未找到" not in generation  # 不是 not_useful_fallback 那套文案
        # 不再输出旧版"原样保留最后一次生成"的行为特征
        assert "该回答经多次核查" not in generation

    def test_无parent_contexts_web路径回退用documents本身(self):
        state = {
            "documents": [
                Document(page_content="网页内容B", metadata={"reranker_score": 0.5, "source": "http://example.com/b"}),
            ],
            "parent_contexts": [],
        }
        result = hallucination_fallback(state)
        generation = result["generation"]

        assert "网页内容B" in generation
        assert "http://example.com/b" in generation

    def test_parent_contexts中的doc_id在documents里查不到时用未知来源兜底(self):
        state = {
            "documents": [],
            "parent_contexts": [
                {"doc_id": "doc-missing", "content": "父块内容C"},
            ],
        }
        result = hallucination_fallback(state)
        generation = result["generation"]

        assert "父块内容C" in generation
        assert "未知来源" in generation

    def test_不再输出state中的generation字段内容(self):
        # 旧版行为是把 state["generation"]（模型自己判定不可靠的答案）原样保留，
        # 新版应该完全不用这个字段
        state = {
            "generation": "这是模型编造的不可靠答案",
            "documents": [Document(page_content="真实检索内容", metadata={"doc_id": "doc-1", "source": "a.md"})],
            "parent_contexts": [{"doc_id": "doc-1", "content": "真实检索内容"}],
        }
        result = hallucination_fallback(state)
        assert "这是模型编造的不可靠答案" not in result["generation"]
