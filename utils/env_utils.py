import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

MILVUS_URI = 'http://localhost:19530'

COLLECTION_NAME = 't_collection01'

POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://raguser:ragpass@localhost:5432/ragdb")

BG_QUALITY_SAMPLE_RATE = float(os.getenv("BG_QUALITY_SAMPLE_RATE", "0.1"))

# 部署级开关：是否允许知识库未覆盖时降级到 Tavily 网络搜索。默认 true（保持现有行为）。
# 纯内部专有知识库场景应设为 false——网上搜到的通用做法可能和内部规范矛盾，
# "未找到"比"拿通用做法冒充内部规范回答"更安全（见 CLAUDE.md 二十八节 web 阈值标定部分的讨论）。
# 进程启动时读一次，不支持单次请求覆盖；改配置需重启服务生效。
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"