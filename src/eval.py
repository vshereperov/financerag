import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from .config import settings
from .ingest import DOCS
from .retrieve import retrieve
from .generate import generate_answer, build_context
from .judge import correctness, faithfulness
from . import usage

K = 5  # Number of retrieved chunks passed to the LLM
GOLD_PAGE_OFFSET = 1  # FinanceBench pages are 0-indexed, but ingested pages are 1-indexed
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


def evaluate(debug=False):
    """Run the full evaluation loop and print retrieval and answer quality metrics."""
    usage.reset()
    items = load_eval_set()
    hits = 0
    latencies = []
    hit_by_type = defaultdict(lambda: {"hits": 0, "total": 0})
    correctness_verdicts = defaultdict(int)
    correctness_by_type = defaultdict(lambda: defaultdict(int))
    correctness_sum = 0.0
    faithfulness_verdicts = defaultdict(int)
    faithfulness_by_type = defaultdict(lambda: defaultdict(int))
    faithfulness_sum = 0.0

    for item in items:
        t0 = time.perf_counter()
        points = retrieve(item["question"], k=K)
        latencies.append(time.perf_counter() - t0)

        hit = is_hit(item, points)
        hits += hit
        hit_by_type[item["question_type"]]["total"] += 1
        hit_by_type[item["question_type"]]["hits"] += hit

        context = build_context(points)
        answer = generate_answer(item["question"], points)

        correctness_verdict = correctness(
            item["question"], item["gold_answer"], answer
        )["verdict"]
        correctness_verdicts[correctness_verdict] += 1
        correctness_by_type[item["question_type"]][correctness_verdict] += 1
        correctness_sum += CORRECTNESS_SCORE.get(correctness_verdict, 0.0)

        faithfulness_verdict = faithfulness(context, answer)["verdict"]
        faithfulness_verdicts[faithfulness_verdict] += 1
        faithfulness_by_type[item["question_type"]][faithfulness_verdict] += 1
        faithfulness_sum += FAITHFULNESS_SCORE.get(faithfulness_verdict, 0.0)

        if debug:
            print(f"[{correctness_verdict:9s}|{faithfulness_verdict:11s}] {item['doc_name']:24s} {item['question'][:45]}")

    n = len(items)
    latencies.sort()

    def section(title):
        print(f"\n{'=' * 60}\n {title}\n{'=' * 60}")

    types = sorted(set(hit_by_type) | set(correctness_by_type) | set(faithfulness_by_type))

    def by_type_verdicts(by_type, labels):
        """Print a per-type breakdown of verdict counts (e.g. correct / partial / incorrect)."""
        print(f"  by type ({' / '.join(labels)}):")
        for question_type in types:
            v = by_type[question_type]
            counts = " / ".join(str(v.get(label, 0)) for label in labels)
            print(f"    {question_type:18s} {counts}")

    section("RETRIEVAL")
    print(f"  Hit-rate @k={K}   {hits}/{n} = {hits / n:.1%}")
    print(f"  Latency (median)  {latencies[n // 2] * 1000:.0f} ms")
    print("  by type:")
    for question_type in types:
        c = hit_by_type.get(question_type)
        if c:
            print(f"    {question_type:18s} {c['hits']}/{c['total']} = {c['hits'] / c['total']:.1%}")

    section("CORRECTNESS")
    cv = correctness_verdicts
    print(f"  Overall  {correctness_sum:.1f}/{n} = {correctness_sum / n:.1%}   "
          f"(correct {cv.get('correct', 0)} / partial {cv.get('partial', 0)} / incorrect {cv.get('incorrect', 0)})")
    by_type_verdicts(correctness_by_type, ["correct", "partial", "incorrect"])

    section("FAITHFULNESS")
    fv = faithfulness_verdicts
    print(f"  Overall  {faithfulness_sum:.1f}/{n} = {faithfulness_sum / n:.1%}   "
          f"(supported {fv.get('supported', 0)} / partial {fv.get('partial', 0)} / unsupported {fv.get('unsupported', 0)})")
    by_type_verdicts(faithfulness_by_type, ["supported", "partial", "unsupported"])

    section("TOKEN USAGE / APPROX COST")
    usage.report()


if __name__ == "__main__":
    evaluate(debug="--debug" in sys.argv)
