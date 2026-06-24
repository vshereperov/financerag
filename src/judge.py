import json
from openai import OpenAI
from .config import settings
from . import usage

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)

CORRECTNESS_SYSTEM_PROMPT = (
    "You are grading a financial-QA system. Questions are answered from company "
    "financial filings and reports (10-K, 10-Q, earnings releases, and similar). "
    "Decide whether the GENERATED answer matches the GOLD answer to the QUESTION on "
    "the substance that matters.\n\n"
    "Reason step by step in `reason` BEFORE choosing a verdict:\n"
    "1. Identify what the gold answer asserts. Depending on the question this may be a "
    "yes/no or which/what conclusion, one or more specific facts (a segment, a driver, "
    "a trend), and/or a figure. Analytical questions hinge on the conclusion and its "
    "main supporting point(s); numeric questions hinge on the figure.\n"
    "2. Check the generated answer against each: does it reach the same conclusion, "
    "cover the same key facts, and — where a figure is central — match it?\n\n"
    "Verdict rule:\n"
    "- correct: same conclusion and key facts/figures as the gold. Equivalent units "
    "($1,577M = $1.577 billion) and minor rounding count as a match. The answer may "
    "use different supporting detail or reasoning than the gold.\n"
    "- partial: right direction or conclusion but missing/imprecise on a key "
    "supporting fact or figure, or only some of several required items are given.\n"
    "- incorrect: the conclusion is wrong or contradicts the gold, a central fact or "
    "figure is wrong, or the answer does not answer / claims it cannot find "
    "information the gold provides.\n\n"
    "Grade substance only: ignore wording, formatting, ordering, and extra "
    "explanation, and never reward length. Respond with ONLY a JSON object, reason "
    'first: {"reason": "<one or two sentences naming what matched or differed>", '
    '"verdict": "correct" | "partial" | "incorrect"}'
)

FAITHFULNESS_SYSTEM_PROMPT = (
    "You check whether an ANSWER drawn from company financial filings and reports "
    "(10-K, 10-Q, earnings releases, and similar) is grounded in the provided "
    "CONTEXT. Judge grounding ONLY, never correctness: a wrong-but-grounded answer is "
    "still supported, and a right-but-unsupported answer is not.\n\n"
    "Reason step by step in `reason` BEFORE choosing a verdict:\n"
    "1. List the factual claims the answer makes (figures, named items, assertions).\n"
    "2. Check each claim against the context: is it stated there, or arithmetically "
    "or logically derivable from figures in the context (e.g. a ratio computed from "
    "context numbers)?\n\n"
    "Verdict rule:\n"
    "- supported: every claim is backed by the context. If the answer says it cannot "
    "find the information, treat it as supported (it makes no claim).\n"
    "- partial: the main claims are backed, but at least one secondary claim is not "
    "in the context.\n"
    "- unsupported: a central claim is absent from or contradicts the context (e.g. a "
    "fabricated figure).\n\n"
    "Do not penalize the answer for omitting information that is in the context, nor "
    "for minor connecting phrases, rephrasings, or obvious restatements of it. General "
    "domain principles (e.g. what a financial ratio means) are not unsupported claims, "
    "but any company-specific fact or figure must come from the context. Flag only "
    "added company-specific claims the context does not support. Respond with ONLY a "
    "JSON object, reason first: "
    '{"reason": "<one or two sentences on which claims are or are not grounded>", '
    '"verdict": "supported" | "partial" | "unsupported"}'
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
            {"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nGOLD: {gold}\n\nGENERATED: {generated}",
            },
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    usage.record(settings.judge_model, response.usage)
    return _parse(response.choices[0].message.content, "incorrect")


def faithfulness(context, answer):
    """Judge whether the answer is grounded in the provided context."""
    response = client.chat.completions.create(
        model=settings.judge_model,
        messages=[
            {"role": "system", "content": FAITHFULNESS_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    usage.record(settings.judge_model, response.usage)
    return _parse(response.choices[0].message.content, "unsupported")
