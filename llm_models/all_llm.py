from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from utils.env_utils import DEEPSEEK_API_KEY

llm = ChatOpenAI(
    temperature=0,
    model='deepseek-chat',
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com")


# max_results=5：走到 web 路径说明知识库已经确认没有覆盖，这时候的主要风险是
# "没东西可用"而不是"垃圾太多"（误判代价不对称，见 graph2/web_search_node.py 的
# TOP_SEGMENTS_PER_RESULT 注释），应该宽进——多几个不同来源，交叉印证的机会更大。
# 增量特意加在"更多来源"而不是"同一篇文章切更多段"：Tavily 一次 API 调用的计费/延迟
# 量级不随返回条数明显变化，比在单篇文章里多切一段划算得多。
web_search_tool = TavilySearchResults(max_results=5)