from collections import defaultdict

_usage = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0, "cost": 0.0})


def record(model, usage_obj):
    """Record token counts and billed cost from a usage object."""
    if usage_obj is None:
        return
    stats = _usage[model]
    stats["input"] += getattr(usage_obj, "prompt_tokens", 0) or 0
    stats["output"] += getattr(usage_obj, "completion_tokens", 0) or 0
    stats["cost"] += getattr(usage_obj, "cost", 0.0) or 0.0
    stats["calls"] += 1


def reset():
    _usage.clear()


def report():
    """Print recorded token usage and billed cost per model."""
    total = 0.0
    for model, stats in _usage.items():
        total += stats["cost"]
        print(
            f"  {model:32s} calls={stats['calls']:3d} "
            f"input={stats['input']:7d} output={stats['output']:6d}  ~${stats['cost']:.4f}"
        )
    print(f"  TOTAL ~${total:.4f}")
