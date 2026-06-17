import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from .config import FINANCEBENCH_DIR
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
JSONL = Path(FINANCEBENCH_DIR) / "data" / "financebench_open_source.jsonl"


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
        faithfulness_sum += FAITHFULNESS_SCORE.get(faithfulness_verdict, 0.0)

        if debug:
            print(f"[{correctness_verdict:9s}|{faithfulness_verdict:11s}] {item['doc_name']:22s} {item['question'][:45]}")

    n = len(items)
    latencies.sort()

    print(f"\nRetrieval hit-rate @k={K}: {hits}/{n} = {hits / n:.1%}")
    print(f"Retrieval latency: median {latencies[n // 2] * 1000:.0f} ms")
    print(f"\nAnswer correctness: {correctness_sum:.1f}/{n} = {correctness_sum / n:.1%}  {dict(correctness_verdicts)}")
    print(f"Faithfulness:       {faithfulness_sum:.1f}/{n} = {faithfulness_sum / n:.1%}  {dict(faithfulness_verdicts)}")
    print("\nCorrectness by type (correct / partial / incorrect):")
    for question_type in sorted(correctness_by_type):
        type_verdicts = correctness_by_type[question_type]
        print(f"  {question_type:20s} {type_verdicts.get('correct', 0)} / {type_verdicts.get('partial', 0)} / {type_verdicts.get('incorrect', 0)}")
    print("\nHit-rate by type:")
    for question_type, counts in sorted(hit_by_type.items()):
        print(f"  {question_type:20s} {counts['hits']}/{counts['total']} = {counts['hits'] / counts['total']:.1%}")

    usage.report()


if __name__ == "__main__":
    evaluate(debug="--debug" in sys.argv)
