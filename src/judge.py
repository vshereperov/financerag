import json
from openai import OpenAI
from .config import settings
from . import usage

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)

CORRECTNESS_SYSTEM = (
    "You are grading a financial QA system. Compare the GENERATED answer to the "
    "GOLD answer for the QUESTION. Judge only whether the generated answer conveys "
    "the same key facts and figures as the gold answer. Ignore differences in "
    "wording, formatting, or extra explanation. A numeric answer must match the gold "
    "figure, allowing equivalent units (e.g. $1,577M = $1.577 billion). Respond with "
    'ONLY a JSON object: {"verdict": "correct" | "partial" | "incorrect", '
    '"reason": "<one short sentence>"}'
)

FAITHFULNESS_SYSTEM = (
    "You check whether an ANSWER is grounded in the provided CONTEXT. Judge only "
    "grounding, not correctness: is every factual claim in the answer supported by "
    "the context? If the answer states it cannot find the information, treat that as "
    "supported (it makes no unsupported claim). Respond with ONLY a JSON object: "
    '{"verdict": "supported" | "partial" | "unsupported", "reason": "<one short sentence>"}'
)


def _parse(text, fallback_verdict):
    """Parse the judge's output. If it is not valid JSON, return a fallback verdict with a reason."""
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "verdict": fallback_verdict,
            "reason": f"unparseable judge output: {text[:80]}",
        }


def correctness(question, gold, generated):
    """Judge whether the generated answer is correct relative to the gold answer."""
    response = client.chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": CORRECTNESS_SYSTEM},
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nGOLD: {gold}\n\nGENERATED: {generated}",
            },
        ],
        temperature=0.0,
    )
    usage.record(settings.judge_model, response.usage)
    return _parse(response.choices[0].message.content, "incorrect")


def faithfulness(context, answer):
    """Judge whether the answer is grounded in the provided context."""
    response = client.chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": FAITHFULNESS_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
        temperature=0.0,
    )
    usage.record(settings.judge_model, response.usage)
    return _parse(response.choices[0].message.content, "unsupported")
