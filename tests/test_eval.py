import json
from types import SimpleNamespace

import pytest

from src import eval as eval_mod


def point(doc_name, page):
    """Stand in for a Qdrant point as retrieve() returns it."""
    return SimpleNamespace(
        payload={
            "company": "AMD",
            "doc_name": doc_name,
            "page": page,
            "content": f"{doc_name} page {page}",
            "summary": "s",
        }
    )


def item(doc_name="AMD_2022_10K", gold_pages=(41,), question_type="metrics-generated"):
    return {
        "id": "financebench_id_0001",
        "question": "What were total assets?",
        "gold_answer": "$67,580 million",
        "question_type": question_type,
        "doc_name": doc_name,
        "gold_pages": list(gold_pages),
    }


# is_hit: the off-by-one that silently shifts every hit-rate we report


def test_is_hit_matches_a_gold_page_across_the_page_numbering_offset():
    """FinanceBench pages are 0-based, ours are 1-based: gold 41 is our page 42."""
    assert eval_mod.is_hit(item(gold_pages=[41]), [point("AMD_2022_10K", 42)])


def test_is_hit_rejects_the_neighbouring_pages():
    """Guards against the offset being dropped (41) or applied twice (43)."""
    assert not eval_mod.is_hit(item(gold_pages=[41]), [point("AMD_2022_10K", 41)])
    assert not eval_mod.is_hit(item(gold_pages=[41]), [point("AMD_2022_10K", 43)])


def test_is_hit_requires_the_right_document():
    """The same page number in another filing is not a hit."""
    assert not eval_mod.is_hit(item(gold_pages=[41]), [point("3M_2023Q2_10Q", 42)])


def test_is_hit_finds_the_gold_page_anywhere_in_the_retrieved_list():
    points = [point("AMD_2022_10K", p) for p in (7, 19, 42, 55)]

    assert eval_mod.is_hit(item(gold_pages=[41]), points)


def test_is_hit_accepts_any_of_several_gold_pages():
    assert eval_mod.is_hit(item(gold_pages=[10, 41]), [point("AMD_2022_10K", 42)])


def test_is_hit_is_false_when_nothing_was_retrieved():
    assert not eval_mod.is_hit(item(), [])


# _unique_base: protects a previous run's results from being overwritten


def test_unique_base_returns_the_name_itself_when_nothing_exists(tmp_path):
    assert eval_mod._unique_base(tmp_path / "results") == tmp_path / "results"


def test_unique_base_steps_past_an_existing_run(tmp_path):
    (tmp_path / "results.txt").write_text("previous run")

    assert eval_mod._unique_base(tmp_path / "results") == tmp_path / "results_2"


def test_unique_base_steps_past_a_json_only_run(tmp_path):
    """Either half of a saved run is enough to make the name taken."""
    (tmp_path / "results.json").write_text("{}")

    assert eval_mod._unique_base(tmp_path / "results") == tmp_path / "results_2"


def test_unique_base_keeps_counting_while_names_are_taken(tmp_path):
    (tmp_path / "results.txt").write_text("run 1")
    (tmp_path / "results_2.txt").write_text("run 2")

    assert eval_mod._unique_base(tmp_path / "results") == tmp_path / "results_3"


# load_eval_set


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)
        f.write("\n")
    return path


def benchmark_record(financebench_id="fb_1", evidence_pages=(41,)):
    return {
        "financebench_id": financebench_id,
        "question": "What were total assets?",
        "answer": "$67,580 million",
        "question_type": "metrics-generated",
        "doc_name": "AMD_2022_10K",
        "company": "AMD",
        "evidence": [{"evidence_page_num": p} for p in evidence_pages],
    }


def test_load_eval_set_reads_the_fields_the_eval_needs(tmp_path, monkeypatch):
    jsonl = write_jsonl(
        tmp_path / "b.jsonl", [benchmark_record(evidence_pages=[41, 42])]
    )
    monkeypatch.setattr(eval_mod, "JSONL", jsonl)

    items = eval_mod.load_eval_set()

    assert items == [
        {
            "id": "fb_1",
            "question": "What were total assets?",
            "gold_answer": "$67,580 million",
            "question_type": "metrics-generated",
            "doc_name": "AMD_2022_10K",
            "gold_pages": [41, 42],
        }
    ]


def test_load_eval_set_skips_blank_lines(tmp_path, monkeypatch):
    jsonl = write_jsonl(
        tmp_path / "b.jsonl", [benchmark_record("fb_1"), benchmark_record("fb_2")]
    )
    monkeypatch.setattr(eval_mod, "JSONL", jsonl)

    assert len(eval_mod.load_eval_set()) == 2


# evaluate: the aggregation behind every number we make decisions on


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace retrieval, generation and both judges so evaluate() runs offline.

    Verdicts are driven per question id, so a test can set up a known mix and
    assert on the scores that come out.
    """

    def install(verdicts, retrieved_page=42):
        monkeypatch.setattr(
            eval_mod, "retrieve", lambda q, k: [point("AMD_2022_10K", retrieved_page)]
        )
        monkeypatch.setattr(eval_mod, "build_context", lambda points: "CONTEXT")
        monkeypatch.setattr(eval_mod, "generate_answer", lambda q, points: "ANSWER")

        state = {"i": 0}

        def next_verdicts():
            cv, fv = verdicts[state["i"]]
            state["i"] += 1
            return cv, fv

        pending = {}

        def fake_correctness(question, gold, generated):
            cv, fv = next_verdicts()
            pending["faithfulness"] = fv
            return {"verdict": cv, "reason": "r"}

        def fake_faithfulness(context, answer):
            return {"verdict": pending["faithfulness"], "reason": "r"}

        monkeypatch.setattr(eval_mod, "correctness", fake_correctness)
        monkeypatch.setattr(eval_mod, "faithfulness", fake_faithfulness)

    return install


def test_evaluate_scores_partial_verdicts_as_half(tmp_path, monkeypatch, stub_pipeline):
    """correct=1, partial=0.5, incorrect=0 -> 1.5/3 = 50%."""
    jsonl = write_jsonl(
        tmp_path / "b.jsonl",
        [benchmark_record(f"fb_{i}") for i in range(3)],
    )
    monkeypatch.setattr(eval_mod, "JSONL", jsonl)
    stub_pipeline(
        [
            ("correct", "supported"),
            ("partial", "partial"),
            ("incorrect", "unsupported"),
        ]
    )

    eval_mod.evaluate(base=str(tmp_path / "results"))

    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    metrics = report["metrics"]
    assert metrics["n"] == 3
    assert metrics["correctness"] == pytest.approx(0.5)
    assert metrics["faithfulness"] == pytest.approx(0.5)
    assert metrics["correctness_verdicts"] == {
        "correct": 1,
        "partial": 1,
        "incorrect": 1,
    }


def test_evaluate_hit_rate_counts_only_retrieved_gold_pages(
    tmp_path, monkeypatch, stub_pipeline
):
    """Two questions, both answered from a page that is nobody's gold page."""
    jsonl = write_jsonl(
        tmp_path / "b.jsonl",
        [benchmark_record("fb_0"), benchmark_record("fb_1")],
    )
    monkeypatch.setattr(eval_mod, "JSONL", jsonl)
    stub_pipeline([("correct", "supported")] * 2, retrieved_page=999)

    eval_mod.evaluate(base=str(tmp_path / "results"))

    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert report["metrics"]["hit_rate"] == 0.0
    assert all(r["hit"] is False for r in report["results"])


def test_evaluate_writes_both_report_files_without_clobbering_a_previous_run(
    tmp_path, monkeypatch, stub_pipeline
):
    jsonl = write_jsonl(tmp_path / "b.jsonl", [benchmark_record()])
    monkeypatch.setattr(eval_mod, "JSONL", jsonl)
    stub_pipeline([("correct", "supported")])
    (tmp_path / "results.txt").write_text("a previous run")

    eval_mod.evaluate(base=str(tmp_path / "results"))

    assert (tmp_path / "results.txt").read_text() == "a previous run"
    assert (tmp_path / "results_2.txt").exists()
    assert (tmp_path / "results_2.json").exists()


def test_evaluate_records_the_answer_and_retrieved_pages_per_question(
    tmp_path, monkeypatch, stub_pipeline
):
    jsonl = write_jsonl(tmp_path / "b.jsonl", [benchmark_record("fb_0")])
    monkeypatch.setattr(eval_mod, "JSONL", jsonl)
    stub_pipeline([("correct", "supported")])

    eval_mod.evaluate(base=str(tmp_path / "results"))

    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    result = report["results"][0]
    assert result["id"] == "fb_0"
    assert result["answer"] == "ANSWER"
    assert result["retrieved"] == [{"doc_name": "AMD_2022_10K", "page": 42}]
    assert result["hit"] is True
