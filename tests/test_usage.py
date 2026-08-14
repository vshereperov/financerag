from types import SimpleNamespace

from src import usage


def make_usage(
    prompt_tokens: int | None = 0,
    completion_tokens: int | None = 0,
    cost: float | None = 0.0,
):
    """Stand in for the usage object OpenAI returns on a response."""
    return SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost
    )


def test_record_stores_tokens_and_cost():
    usage.record("deepseek/deepseek-v4-flash", make_usage(100, 20, 0.001))

    stats = usage._usage["deepseek/deepseek-v4-flash"]
    assert stats == {"input": 100, "output": 20, "calls": 1, "cost": 0.001}


def test_record_accumulates_across_calls():
    usage.record("gpt-5.1", make_usage(10, 5, 0.01))
    usage.record("gpt-5.1", make_usage(30, 15, 0.02))

    stats = usage._usage["gpt-5.1"]
    assert stats["input"] == 40
    assert stats["output"] == 20
    assert stats["calls"] == 2
    assert stats["cost"] == 0.03


def test_record_ignores_missing_usage():
    usage.record("model-a", None)

    assert usage._usage == {}


def test_record_treats_missing_fields_as_zero():
    """Not every provider returns every field: a bare or None-valued usage object
    must not blow up the run or poison the cost total."""
    usage.record("model-a", SimpleNamespace(prompt_tokens=5))
    usage.record("model-b", make_usage(None, None, None))

    assert usage._usage["model-a"] == {"input": 5, "output": 0, "calls": 1, "cost": 0.0}
    assert usage._usage["model-b"] == {"input": 0, "output": 0, "calls": 1, "cost": 0.0}
