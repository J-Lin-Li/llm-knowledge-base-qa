"""
测试 transform_query_node.py 的计数器隔离逻辑。

核心设计决策：两条路进同一个节点，但各自只动自己的计数器：
  - 有文档（not_useful 触发）→ 只增 not_useful_count，不动 transform_count
  - 无文档（无文档路触发）  → 只增 transform_count，不动 not_useful_count
  两条路都会把 generation_count 清零（幻觉计数隔离）

测试技术说明：
  @patch('graph2.transform_query_node.llm', new=RunnableLambda(lambda x: "改写后的问题"))

  transform_query 内部会用 llm 构造链并调用 LLM 改写问题。
  我们不关心改写结果，只关心 return dict 里的计数器。
  用 RunnableLambda 替换 llm，让链能正常执行但直接返回固定字符串，
  避免真实 LLM 调用，同时保证 StrOutputParser 能收到合法的字符串输入。

  为什么不用 MagicMock 替换 llm？
  因为 MagicMock 不是 Runnable，LangChain 会把它包成 RunnableLambda(MagicMock)，
  调用时 MagicMock() 返回 MagicMock 对象，StrOutputParser 接收后可能触发 Pydantic
  校验失败。RunnableLambda 直接返回字符串，链路干净。
"""

import pytest
from unittest.mock import patch
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from graph2.transform_query_node import transform_query

# 替换 llm 用的假 Runnable，接收任意输入，返回固定改写结果
MOCK_LLM = RunnableLambda(lambda x: "改写后的问题（测试用）")


class TestTransformQueryCounterIsolation:
    """验证两条触发路径的计数器完全隔离"""

    @patch('graph2.transform_query_node.llm', new=MOCK_LLM)
    def test_有文档时只增not_useful_count(self):
        state = {
            "question": "Milvus如何配置索引",
            "documents": [Document(page_content="文档内容")],
            "transform_count": 0,
            "not_useful_count": 0,
            "generation_count": 2,
        }
        result = transform_query(state)

        assert result["not_useful_count"] == 1    # 只有这个 +1
        assert "transform_count" not in result    # 不应出现在返回 dict 里

    @patch('graph2.transform_query_node.llm', new=MOCK_LLM)
    def test_无文档时只增transform_count(self):
        state = {
            "question": "Milvus如何配置索引",
            "documents": [],
            "transform_count": 0,
            "not_useful_count": 0,
            "generation_count": 1,
        }
        result = transform_query(state)

        assert result["transform_count"] == 1     # 只有这个 +1
        assert "not_useful_count" not in result   # 不应出现在返回 dict 里

    @patch('graph2.transform_query_node.llm', new=MOCK_LLM)
    def test_有文档时generation_count清零(self):
        state = {
            "question": "问题",
            "documents": [Document(page_content="文档")],
            "transform_count": 0,
            "not_useful_count": 0,
            "generation_count": 2,  # 已经重试过幻觉了
        }
        result = transform_query(state)

        # 进入新一轮检索+生成，幻觉计数从头开始
        assert result["generation_count"] == 0

    @patch('graph2.transform_query_node.llm', new=MOCK_LLM)
    def test_无文档时generation_count清零(self):
        state = {
            "question": "问题",
            "documents": [],
            "transform_count": 0,
            "not_useful_count": 0,
            "generation_count": 1,
        }
        result = transform_query(state)

        assert result["generation_count"] == 0


class TestTransformQueryCounterAccumulation:
    """验证多次调用时计数器正确累加"""

    @patch('graph2.transform_query_node.llm', new=MOCK_LLM)
    def test_not_useful路径计数正确累加(self):
        # 第一次改写
        state = {
            "question": "问题",
            "documents": [Document(page_content="文档")],
            "transform_count": 0,
            "not_useful_count": 0,
            "generation_count": 0,
        }
        result = transform_query(state)
        assert result["not_useful_count"] == 1

        # 模拟第二次（用上一次的结果继续）
        state["not_useful_count"] = result["not_useful_count"]
        result2 = transform_query(state)
        assert result2["not_useful_count"] == 2

    @patch('graph2.transform_query_node.llm', new=MOCK_LLM)
    def test_无文档路径计数正确累加(self):
        state = {
            "question": "问题",
            "documents": [],
            "transform_count": 1,  # 已经改写过一次
            "not_useful_count": 0,
            "generation_count": 0,
        }
        result = transform_query(state)
        assert result["transform_count"] == 2  # 再 +1 变成 2，下次触发降级
