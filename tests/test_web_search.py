"""
测试 web_search 路径的新增逻辑：切分、精排、阈值过滤。

被测函数：
  - _split_if_long        ：纯函数，超长切分
  - web_search             ：节点函数，调用 web_search_tool + reranker（都需要 mock）
  - grade_web_results      ：节点函数，纯阈值比较（依赖模块级 WEB_RERANKER_THRESHOLD）
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import graph2.web_search_node as web_search_module
import graph2.grade_web_results_node as grade_web_module
from graph2.web_search_node import web_search, _split_if_long, WEB_SPLIT_CHARS
from graph2.grade_web_results_node import grade_web_results


# ═══════════════════════════════════════════════════════
# _split_if_long — 纯函数
# ═══════════════════════════════════════════════════════

class TestSplitIfLong:

    def test_短内容不切分(self):
        content = "短内容"
        assert _split_if_long(content) == [content]

    def test_刚好等于阈值不切分(self):
        content = "字" * WEB_SPLIT_CHARS
        assert _split_if_long(content) == [content]

    def test_超长内容按定长切分且不丢字(self):
        content = "字" * (WEB_SPLIT_CHARS * 2 + 10)
        segments = _split_if_long(content)
        assert len(segments) == 3
        assert segments[0] == content[:WEB_SPLIT_CHARS]
        assert "".join(segments) == content


# ═══════════════════════════════════════════════════════
# web_search 节点 — 需要 mock web_search_tool 和 reranker
# ═══════════════════════════════════════════════════════

class TestWebSearchNode:

    def test_每条结果只保留top1段(self, monkeypatch):
        raw_results = [
            {"url": "http://a.com", "content": "内容A"},
            {"url": "http://b.com", "content": "内容B"},
        ]
        monkeypatch.setattr(web_search_module.web_search_tool, "invoke", MagicMock(return_value=raw_results))
        with patch.object(web_search_module.reranker, "predict", return_value=[0.5]):
            result = web_search({"question": "测试问题"})

        docs = result["documents"]
        assert len(docs) == 2  # 每条结果1段（TOP_SEGMENTS_PER_RESULT=1），2条结果
        assert docs[0].metadata["source"] == "http://a.com"
        assert docs[1].metadata["source"] == "http://b.com"
        assert all("reranker_score" in d.metadata for d in docs)

    def test_超长结果切分后只保留最高分那段(self, monkeypatch):
        long_content = "A" * WEB_SPLIT_CHARS + "B" * WEB_SPLIT_CHARS  # 切成2段
        raw_results = [{"url": "http://a.com", "content": long_content}]
        monkeypatch.setattr(web_search_module.web_search_tool, "invoke", MagicMock(return_value=raw_results))

        # 第二段（B开头）分数更高，应该被保留，第一段丢弃
        with patch.object(web_search_module.reranker, "predict", return_value=[0.1, 0.9]):
            result = web_search({"question": "测试问题"})

        docs = result["documents"]
        assert len(docs) == 1
        assert docs[0].page_content.startswith("B")
        assert docs[0].metadata["reranker_score"] == 0.9

    def test_无搜索结果_返回空列表(self, monkeypatch):
        monkeypatch.setattr(web_search_module.web_search_tool, "invoke", MagicMock(return_value=[]))
        result = web_search({"question": "测试问题"})
        assert result["documents"] == []

    def test_不传config也能正常运行(self, monkeypatch):
        raw_results = [{"url": "http://a.com", "content": "内容"}]
        monkeypatch.setattr(web_search_module.web_search_tool, "invoke", MagicMock(return_value=raw_results))
        with patch.object(web_search_module.reranker, "predict", return_value=[0.5]):
            result = web_search({"question": "测试问题"}, config=None)
        assert len(result["documents"]) == 1


# ═══════════════════════════════════════════════════════
# grade_web_results 节点 — 纯阈值比较，依赖模块级 WEB_RERANKER_THRESHOLD（默认 None）
# ═══════════════════════════════════════════════════════

class TestGradeWebResults:

    def _doc(self, score):
        return Document(page_content="内容", metadata={"reranker_score": score})

    def test_阈值未标定时使用_显式报错而非静默瞎跑(self, monkeypatch):
        # 不再依赖模块默认值恰好是 None——WEB_RERANKER_THRESHOLD 已标定为 0.1
        # （见 CLAUDE.md 二十一节），这里显式 patch 成 None 来测"未标定时报错"
        # 这条代码路径本身还在，和当前实际标定成什么值解耦。
        monkeypatch.setattr(grade_web_module, "WEB_RERANKER_THRESHOLD", None)
        state = {"question": "q", "documents": [self._doc(0.5)]}
        with pytest.raises(RuntimeError):
            grade_web_results(state)

    def test_按阈值过滤(self, monkeypatch):
        monkeypatch.setattr(grade_web_module, "WEB_RERANKER_THRESHOLD", -0.5)
        state = {"question": "q", "documents": [self._doc(-1.0), self._doc(0.0)]}
        result = grade_web_results(state)
        assert len(result["documents"]) == 1
        assert result["documents"][0].metadata["reranker_score"] == 0.0

    def test_reranker_score缺失时默认保留(self, monkeypatch):
        monkeypatch.setattr(grade_web_module, "WEB_RERANKER_THRESHOLD", 0.3)
        doc_no_score = Document(page_content="内容", metadata={})
        state = {"question": "q", "documents": [doc_no_score]}
        result = grade_web_results(state)
        assert len(result["documents"]) == 1
