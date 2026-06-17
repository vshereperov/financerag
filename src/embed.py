from openai import OpenAI
from .config import OPENROUTER_API_KEY, EMBEDDING_MODEL
from . import usage

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def embed_texts(texts):
    """Embed a list of texts using the OpenRouter API."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    usage.record(EMBEDDING_MODEL, response.usage)
    return [item.embedding for item in response.data]
