from openai import OpenAI
from fastembed import SparseTextEmbedding
from .config import OPENROUTER_API_KEY, EMBEDDING_MODEL, SPARSE_MODEL
from . import usage

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Lazy initialization
_sparse_model = None


def embed_texts(texts):
    """Embed a list of texts using the OpenRouter API."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    usage.record(EMBEDDING_MODEL, response.usage)
    return [item.embedding for item in response.data]


def _sparse():
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    return _sparse_model


def embed_sparse_docs(texts):
    """Compute BM25 sparse vectors for documents (TF weighted)."""
    return list(_sparse().embed(texts))


def embed_sparse_query(text):
    """Compute a BM25 sparse vector for a query (IDF weighted, no TF saturation)."""
    return list(_sparse().query_embed([text]))[0]
