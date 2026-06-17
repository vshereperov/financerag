from openai import OpenAI
from .config import OPENROUTER_API_KEY, LLM_MODEL
from . import usage

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the question using ONLY the "
    "context from company 10-K filings provided below. If the answer is not in "
    "the context, say you cannot find it in the provided documents. Always cite "
    "the source as (company, document, page)."
)


def build_context(points):
    """Build a context string from the retrieved points, including company, document name, page number, and content."""
    blocks = []
    for point in points:
        payload = point.payload
        blocks.append(
            f"[{payload['company']} | {payload['doc_name']} | page {payload['page']}]\n{payload['content']}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question, points):
    """Generate an answer to the question using the retrieved points as context."""
    context = build_context(points)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.0,
    )
    usage.record(LLM_MODEL, response.usage)
    return response.choices[0].message.content
