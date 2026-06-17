import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
FINANCEBENCH_DIR = os.environ["FINANCEBENCH_DIR"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
EMBEDDING_DIM = int(os.environ["EMBEDDING_DIM"])
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]
LLM_MODEL = os.environ["LLM_MODEL"]
JUDGE_MODEL = os.environ["JUDGE_MODEL"]
