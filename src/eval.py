import io
import sys
import json
import time
import contextlib
from pathlib import Path
from collections import defaultdict

from .config import settings
from .ingest import DOCS
from .retrieve import retrieve
from .generate import generate_answer, build_context
from .judge import correctness, faithfulness
from . import usage

GOLD_PAGE_OFFSET = 1  # FinanceBench pages start at 0, ours at 1
CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
FAITHFULNESS_SCORE = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}

INGESTED_DOCS = {doc_name for _, doc_name in DOCS}
JSONL = Path(settings.financebench_dir) / "data" / "financebench_open_source.jsonl"


def load_eval_set():
    """Load and filter FinanceBench questions to only those covered by ingested documents."""
    items = []
    with open(JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["doc_name"] not in INGESTED_DOCS:
                continue
            items.append(
                {
                    "id": record["financebench_id"],
                    "question": record["question"],
                    "gold_answer": record["answer"],
                    "question_type": record["question_type"],
                    "doc_name": record["doc_name"],
                    "gold_pages": [e["evidence_page_num"] for e in record["evidence"]],
                }
            )
    return items


def is_hit(item, points):
    """Return True if any retrieved point covers one of the item's gold pages."""
    for point in points:
        payload = point.payload
        if payload["doc_name"] != item["doc_name"]:
            continue
        for gold_page in item["gold_pages"]:
            if payload["page"] == gold_page + GOLD_PAGE_OFFSET:
                return True
    return False


def _unique_base(base_path):
    """Avoid overwriting a previous run's results by finding a free output name."""
    candidate = base_path
    i = 2
    while (
        candidate.with_suffix(".txt").exists()
        or candidate.with_suffix(".json").exists()
    ):
        candidate = base_path.with_name(f"{base_path.name}_{i}")
        i += 1
    return candidate


def evaluate(debug=False, base="evals/results"):
    """Run the full eval and save the report to <base>.txt and a detailed JSON record to <base>.json."""
    usage.reset()
    items = load_eval_set()
    n = len(items)
    hits = 0
    latencies = []
    hit_by_type = defaultdict(lambda: {"hits": 0, "total": 0})
    correctness_verdicts = defaultdict(int)
    correctness_by_type = defaultdict(lambda: defaultdict(int))
    correctness_sum = 0.0
    faithfulness_verdicts = defaultdict(int)
    faithfulness_by_type = defaultdict(lambda: defaultdict(int))
    faithfulness_sum = 0.0
    results = []  # per-question records for the JSON artifact

    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        for item in items:
            t0 = time.perf_counter()
            points = retrieve(item["question"], k=settings.top_k)
            latencies.append(time.perf_counter() - t0)

            hit = is_hit(item, points)
            hits += hit
            hit_by_type[item["question_type"]]["total"] += 1
            hit_by_type[item["question_type"]]["hits"] += hit

            context = build_context(points)
            answer = generate_answer(item["question"], points)

            correctness_result = correctness(
                item["question"], item["gold_answer"], answer
            )
            faithfulness_result = faithfulness(context, answer)
            cv = correctness_result["verdict"]
            fv = faithfulness_result["verdict"]

            correctness_verdicts[cv] += 1
            correctness_by_type[item["question_type"]][cv] += 1
            correctness_sum += CORRECTNESS_SCORE.get(cv, 0.0)
            faithfulness_verdicts[fv] += 1
            faithfulness_by_type[item["question_type"]][fv] += 1
            faithfulness_sum += FAITHFULNESS_SCORE.get(fv, 0.0)

            results.append(
                {
                    "id": item["id"],
                    "doc_name": item["doc_name"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "gold_answer": item["gold_answer"],
                    "gold_pages": item["gold_pages"],
                    "answer": answer,
                    "retrieved": [
                        {"doc_name": payload["doc_name"], "page": payload["page"]}
                        for p in points
                        if (payload := p.payload) is not None
                    ],
                    "hit": hit,
                    "correctness": correctness_result,
                    "faithfulness": faithfulness_result,
                }
            )

            if debug:
                print(
                    f"[{'hit ' if hit else 'miss'}|{cv:9s}|{fv:11s}] "
                    f"{item['doc_name']:24s} {item['question'][:45]}"
                )

        latencies.sort()
        median_latency_ms = (latencies[(n - 1) // 2] + latencies[n // 2]) / 2 * 1000

        def section(title):
            print(f"\n{'=' * 60}\n {title}\n{'=' * 60}")

        types = sorted(
            set(hit_by_type) | set(correctness_by_type) | set(faithfulness_by_type)
        )

        def by_type_verdicts(by_type, labels):
            """Print a per-type breakdown of verdict counts."""
            print(f"  by type ({' / '.join(labels)}):")
            for question_type in types:
                v = by_type[question_type]
                counts = " / ".join(str(v.get(label, 0)) for label in labels)
                print(f"    {question_type:18s} {counts}")

        section("RETRIEVAL")
        print(f"  Hit-rate @k={settings.top_k}   {hits}/{n} = {hits / n:.1%}")
        print(f"  Latency (median)  {median_latency_ms:.0f} ms")
        print("  by type:")
        for question_type in types:
            c = hit_by_type.get(question_type)
            if c:
                print(
                    f"    {question_type:18s} {c['hits']}/{c['total']} = {c['hits'] / c['total']:.1%}"
                )

        section("CORRECTNESS")
        cv_counts = correctness_verdicts
        print(
            f"  Overall  {correctness_sum:.1f}/{n} = {correctness_sum / n:.1%}   "
            f"(correct {cv_counts.get('correct', 0)} / partial {cv_counts.get('partial', 0)} / incorrect {cv_counts.get('incorrect', 0)})"
        )
        by_type_verdicts(correctness_by_type, ["correct", "partial", "incorrect"])

        section("FAITHFULNESS")
        fv_counts = faithfulness_verdicts
        print(
            f"  Overall  {faithfulness_sum:.1f}/{n} = {faithfulness_sum / n:.1%}   "
            f"(supported {fv_counts.get('supported', 0)} / partial {fv_counts.get('partial', 0)} / unsupported {fv_counts.get('unsupported', 0)})"
        )
        by_type_verdicts(faithfulness_by_type, ["supported", "partial", "unsupported"])

        section("TOKEN USAGE / COST")
        usage.report()

    output = {
        "metrics": {
            "n": n,
            "hit_rate": hits / n,
            "latency_median_ms": median_latency_ms,
            "hit_by_type": {t: dict(v) for t, v in hit_by_type.items()},
            "correctness": correctness_sum / n,
            "correctness_verdicts": dict(correctness_verdicts),
            "correctness_by_type": {t: dict(v) for t, v in correctness_by_type.items()},
            "faithfulness": faithfulness_sum / n,
            "faithfulness_verdicts": dict(faithfulness_verdicts),
            "faithfulness_by_type": {
                t: dict(v) for t, v in faithfulness_by_type.items()
            },
        },
        "results": results,
    }

    base_path = Path(base).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = _unique_base(base_path)
    txt_path = base_path.with_suffix(".txt")
    json_path = base_path.with_suffix(".json")
    txt_path.write_text(capture.getvalue(), encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved {txt_path} and {json_path}", file=sys.stderr)


if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    base = positional[0] if positional else "evals/results"
    evaluate(debug="--debug" in sys.argv, base=base)
