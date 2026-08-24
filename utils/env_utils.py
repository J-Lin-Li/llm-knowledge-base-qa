import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

MILVUS_URI = 'http://localhost:19530'

COLLECTION_NAME = 't_collection01'

POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://raguser:ragpass@localhost:5432/ragdb")

BG_QUALITY_SAMPLE_RATE = float(os.getenv("BG_QUALITY_SAMPLE_RATE", "0.1"))