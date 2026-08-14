from types import SimpleNamespace

from src import rerank as rerank_mod
from src import usage


def point(page):
    return SimpleNamespace(
        payload={"doc_name": "AMD_2022_10K", "page": page, "content": f"page {page}"}
    )


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class FakeClient:
    """Stands in for the OpenAI client so no request ever leaves the test."""

    def __init__(self, body):
        self.body = body
        self.calls = []

    def post(self, path, **kwargs):
        self.calls.append({"path": path, **kwargs})
        return FakeResponse(self.body)


def api_body(indices, usage_obj=None):
    """The shape the rerank API answers with: results carry indices into our input."""
    body = {"results": [{"index": i, "relevance_score": 1.0} for i in indices]}
    if usage_obj is not None:
        body["usage"] = usage_obj
    return body


def install(monkeypatch, body):
    fake = FakeClient(body)
    monkeypatch.setattr(rerank_mod, "client", fake)
    return fake


# the index mapping: a bug here silently feeds the generator wrong pages


def test_rerank_reorders_points_by_the_indices_the_api_returns(monkeypatch):
    install(monkeypatch, api_body([2, 0, 1]))
    points = [point(10), point(20), point(30)]

    result = rerank_mod.rerank("q", points, k=3)

    assert [p.payload["page"] for p in result] == [30, 10, 20]


def test_rerank_returns_only_the_points_the_api_kept(monkeypatch):
    """top_n=2 of a 4-point pool: the two dropped points must not come back."""
    install(monkeypatch, api_body([3, 1]))
    points = [point(10), point(20), point(30), point(40)]

    result = rerank_mod.rerank("q", points, k=2)

    assert [p.payload["page"] for p in result] == [40, 20]


def test_rerank_returns_the_original_point_objects(monkeypatch):
    """The payload has to survive reranking - the eval reads doc_name/page off it."""
    install(monkeypatch, api_body([1, 0]))
    points = [point(10), point(20)]

    result = rerank_mod.rerank("q", points, k=2)

    assert result[0] is points[1]
    assert result[1] is points[0]


def test_rerank_sends_the_page_contents_and_the_configured_top_n(monkeypatch):
    fake = install(monkeypatch, api_body([0, 1]))
    points = [point(10), point(20)]

    rerank_mod.rerank("what were total assets?", points, k=2)

    sent = fake.calls[0]["body"]
    assert sent["query"] == "what were total assets?"
    assert sent["documents"] == ["page 10", "page 20"]
    assert sent["top_n"] == 2


def test_rerank_skips_the_api_call_on_an_empty_pool(monkeypatch):
    """Retrieval can come back empty; that must not turn into a paid API call."""
    fake = install(monkeypatch, api_body([]))

    assert rerank_mod.rerank("q", [], k=10) == []
    assert fake.calls == []


# cost accounting


def test_rerank_records_the_billed_tokens_and_cost(monkeypatch):
    install(
        monkeypatch,
        api_body([0], usage_obj={"total_tokens": 1200, "cost": 0.004}),
    )

    rerank_mod.rerank("q", [point(10)], k=1)

    stats = usage._usage[rerank_mod.settings.rerank_model]
    assert stats == {"input": 1200, "output": 0, "calls": 1, "cost": 0.004}


def test_rerank_survives_a_response_with_no_usage_block(monkeypatch):
    """Not every provider bills back a usage object; reranking must still work."""
    install(monkeypatch, api_body([0]))

    result = rerank_mod.rerank("q", [point(10)], k=1)

    assert [p.payload["page"] for p in result] == [10]
    assert usage._usage[rerank_mod.settings.rerank_model]["cost"] == 0.0
