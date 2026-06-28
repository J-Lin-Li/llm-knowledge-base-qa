"""
测试 graph_2.py 里的两个决策函数。

这两个函数是整个状态机的"交叉路口"，每一个分支对应一条不同的执行路径，
测不到它们，死循环保护形同虚设。

被测函数：
  - decide_to_generate(state)         ：文档打分后，决定去生成 or 改写 or 网搜
  - grade_generation_v_documents_and_question(state)：生成后，决定结束 or 重试 or 改写

测试技术说明：
  @patch('graph2.graph_2.hallucination_grader_chain')
  这行装饰器的含义：
    - 找到 graph2.graph_2 模块里名为 hallucination_grader_chain 的变量
    - 在这个测试函数运行期间，把它替换成一个 MagicMock
    - 测试函数结束后自动恢复原值
  为什么 patch 模块里的变量而不是 patch 原始定义处？
    因为 graph_2.py 已经执行了 `from ... import hallucination_grader_chain`，
    它拿到的是一个本地引用，patch 原始模块不影响这个已有引用。
    必须 patch 使用它的那个模块里的名字。
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from graph2.graph_2 import decide_to_generate, grade_generation_v_documents_and_question


# ═══════════════════════════════════════════════════════
# decide_to_generate 测试
# 这是纯逻辑函数，只读 state dict，不调用任何外部服务，无需 mock
# ═══════════════════════════════════════════════════════

class TestDecideToGenerate:

    def test_有相关文档_走生成节点(self):
        state = {
            "documents": [Document(page_content="Milvus部署文档内容")],
            "transform_count": 0,
        }
        assert decide_to_generate(state) == "generate"

    def test_无文档且改写未到上限_走查询改写(self):
        # transform_count=1，还没到2，应该继续改写
        state = {
            "documents": [],
            "transform_count": 1,
        }
        assert decide_to_generate(state) == "transform_query"

    def test_无文档且改写刚好到上限_降级网络搜索(self):
        # transform_count=2，触发降级
        state = {
            "documents": [],
            "transform_count": 2,
        }
        assert decide_to_generate(state) == "web_search"

    def test_无文档且改写超过上限_同样降级网络搜索(self):
        # 超过 2 也应该走 web_search，不应崩溃
        state = {
            "documents": [],
            "transform_count": 5,
        }
        assert decide_to_generate(state) == "web_search"

    def test_state没有transform_count字段_视为0(self):
        # state.get("transform_count", 0) 保证缺字段时不崩溃
        state = {"documents": []}
        assert decide_to_generate(state) == "transform_query"


# ═══════════════════════════════════════════════════════
# grade_generation_v_documents_and_question 测试
# 这个函数调用两条外部 chain，用 @patch 替换它们，
# 让我们完全控制 LLM 返回什么，从而测试分支逻辑
# ═══════════════════════════════════════════════════════

class TestGradeGeneration:

    def _state(self, generation_count=0, not_useful_count=0):
        """构造测试用的最小 state，避免每个测试重复写"""
        return {
            "question": "Milvus如何部署？",
            "documents": [Document(page_content="使用Docker Compose部署。")],
            "generation": "使用Docker Compose部署Milvus。",
            "generation_count": generation_count,
            "not_useful_count": not_useful_count,
        }

    # ── 正常通过路径 ──

    # patch(...)
    # 返回的这个装饰器做了三件事：
    # 1.测试函数运行前：把graph2.graph_2模块里的hallucination_grader_chain替换成MagicMock
    # 2.把那个MagicMock作为额外参数注入给测试函数（就是mock_hallucination）
    # 3.测试函数运行后：把原来的hallucination_grader_chain恢复回去


    @patch('graph2.graph_2.hallucination_grader_chain')
    @patch('graph2.graph_2.answer_grader_chain')
    def test_无幻觉且解决了问题_返回useful(self, mock_answer, mock_hallucination):
        # @patch 装饰器从下往上对应参数从左往右
        mock_hallucination.invoke.return_value = MagicMock(binary_score="yes")
        mock_answer.invoke.return_value = MagicMock(binary_score="yes")

        result = grade_generation_v_documents_and_question(self._state())
        assert result == "useful"

    # ── 答案质量失败路径 ──

    @patch('graph2.graph_2.hallucination_grader_chain')
    @patch('graph2.graph_2.answer_grader_chain')
    def test_无幻觉但答案差且未到上限_返回not_useful(self, mock_answer, mock_hallucination):
        mock_hallucination.invoke.return_value = MagicMock(binary_score="yes")
        mock_answer.invoke.return_value = MagicMock(binary_score="no")

        # not_useful_count=1，还没到2，应该继续改写
        result = grade_generation_v_documents_and_question(self._state(not_useful_count=1))
        assert result == "not useful"

    @patch('graph2.graph_2.hallucination_grader_chain')
    @patch('graph2.graph_2.answer_grader_chain')
    def test_无幻觉但答案差且到达上限_返回transform_max_retries(self, mock_answer, mock_hallucination):
        mock_hallucination.invoke.return_value = MagicMock(binary_score="yes")
        mock_answer.invoke.return_value = MagicMock(binary_score="no")

        # not_useful_count=2，触发死循环保护
        result = grade_generation_v_documents_and_question(self._state(not_useful_count=2))
        assert result == "transform_max_retries"

    # ── 幻觉失败路径 ──

    @patch('graph2.graph_2.hallucination_grader_chain')
    def test_有幻觉且未到重试上限_返回not_supported(self, mock_hallucination):
        mock_hallucination.invoke.return_value = MagicMock(binary_score="no")

        # generation_count=2，还没到3，应该重新生成
        result = grade_generation_v_documents_and_question(self._state(generation_count=2))
        assert result == "not supported"

    @patch('graph2.graph_2.hallucination_grader_chain')
    def test_有幻觉且到达重试上限_返回max_retries(self, mock_hallucination):
        mock_hallucination.invoke.return_value = MagicMock(binary_score="no")

        # generation_count=3，触发幻觉死循环保护
        result = grade_generation_v_documents_and_question(self._state(generation_count=3))
        assert result == "max_retries"

    # ── 边界值：上限临界点 ──

    @patch('graph2.graph_2.hallucination_grader_chain')
    def test_幻觉重试上限是3次而非2次(self, mock_hallucination):
        mock_hallucination.invoke.return_value = MagicMock(binary_score="no")

        # generation_count=2（第3次生成后），还应该允许重试
        assert grade_generation_v_documents_and_question(
            self._state(generation_count=2)
        ) == "not supported"

        # generation_count=3（第4次生成后），才触发上限
        assert grade_generation_v_documents_and_question(
            self._state(generation_count=3)
        ) == "max_retries"
