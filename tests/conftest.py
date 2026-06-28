"""
conftest.py 是 pytest 的全局配置文件，在所有测试之前自动执行。

这里做一件关键的事：在任何 project 模块被 import 之前，
把有"import 副作用"的模块替换成 MagicMock。

import 副作用指：模块被 import 时就执行了真实操作，而不只是定义函数。
本项目有三处：
  1. tools/retriever_tools.py  —— import 时连接 Milvus（mv.create_connection()）
  2. llm_models/embeddings_model.py —— import 时加载 BGE 模型（HuggingFaceEmbeddings(...)）
  3. llm_models/all_llm.py  —— import 时初始化 LLM 客户端（ChatOpenAI(...)）

sys.modules 是 Python 的模块缓存字典。
Python import 时先查 sys.modules，找到就直接用，不再执行真实文件。
所以在这里提前塞入 MagicMock，就能完全拦截真实模块的加载。
"""

import sys
from unittest.mock import MagicMock

# ── 必须在任何 project import 之前执行，放在文件顶层即可保证顺序 ──

# Milvus 相关：pymilvus 和 langchain_milvus 都不需要真实连接
sys.modules['pymilvus'] = MagicMock()
sys.modules['langchain_milvus'] = MagicMock()

# BGE 模型：HuggingFaceEmbeddings 加载模型文件，测试不需要真实模型
sys.modules['langchain_huggingface'] = MagicMock()
sys.modules['llm_models.embeddings_model'] = MagicMock(bge_embedding=MagicMock())

# LLM 客户端：ChatOpenAI / TavilySearchResults 初始化需要 API Key，测试不需要真实调用
sys.modules['llm_models.all_llm'] = MagicMock(
    llm=MagicMock(),
    web_search_tool=MagicMock()
)

# retriever_tools：import 时调用 mv.create_connection()，需要 Milvus 在线
sys.modules['tools.retriever_tools'] = MagicMock(
    retriever=MagicMock(),
    retriever_tool=MagicMock()
)
