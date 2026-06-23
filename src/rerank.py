from types import SimpleNamespace

import httpx
from openai import OpenAI

from .config import settings
from . import usage

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)


def rerank(query, points, k):
    """Rerank retrieved points via the OpenRouter rerank API and return the top k."""
    if not points:
        return points
    documents = [p.payload["content"] for p in points]
    response = client.post(
        "/rerank",
        body={
            "query": query,
            "documents": documents,
            "model": settings.rerank_model,
            "top_n": k,
        },
        cast_to=httpx.Response,
    )
    body = response.json()
    billed = body.get("usage") or {}
    usage.record(
        settings.rerank_model,
        SimpleNamespace(
            prompt_tokens=billed.get("total_tokens") or 0,
            cost=billed.get("cost") or 0.0,
        ),
    )
    return [points[r["index"]] for r in body["results"]]
