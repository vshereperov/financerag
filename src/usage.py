from collections import defaultdict

# Prices per 1M tokens: (input, output)
PRICES = {
    "openai/text-embedding-3-small": (0.02, 0.0),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.0),
}

_usage = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})


def record(model, usage_obj):
    """Record token usage for a given model and usage object returned by the OpenRouter API."""
    if usage_obj is None:
        return
    stats = _usage[model]
    stats["input"] += getattr(usage_obj, "prompt_tokens", 0) or 0
    stats["output"] += getattr(usage_obj, "completion_tokens", 0) or 0
    stats["calls"] += 1


def reset():
    _usage.clear()


def report():
    """Print a report of token usage and estimated costs for all recorded models."""
    print("\nToken usage / approx cost:")
    total = 0.0
    for model, stats in _usage.items():
        line = (
            f"  {model:32s} calls={stats['calls']:3d} "
            f"input={stats['input']:7d} output={stats['output']:6d}"
        )
        prices = PRICES.get(model)
        if prices is not None:
            input_price, output_price = prices
            cost = (
                stats["input"] / 1e6 * input_price
                + stats["output"] / 1e6 * output_price
            )
            total += cost
            line += f"  ~${cost:.4f}"
        print(line)
    print(f"  TOTAL ~${total:.4f}")
