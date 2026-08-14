from types import SimpleNamespace

import pytest

from src import retrieve as retrieve_mod


def point(page):
    return SimpleNamespace(payload={"page": page, "content": f"page {page}"})


@pytest.fixture
def pipeline(monkeypatch):
    """Record what the retrieval and rerank stages were asked for, without calling them."""
    calls = {"dense": [], "hybrid": [], "rerank": []}

    def fake_dense(query, k):
        calls["dense"].append({"query": query, "k": k})
        return [point(i) for i in range(k)]

    def fake_hybrid(query, k):
        calls["hybrid"].append({"query": query, "k": k})
        return [point(i) for i in range(k)]

    def fake_rerank(query, points, k):
        calls["rerank"].append({"query": query, "points": points, "k": k})
        return points[:k]

    monkeypatch.setattr(retrieve_mod, "_retrieve_dense", fake_dense)
    monkeypatch.setattr(retrieve_mod, "_retrieve_hybrid", fake_hybrid)
    monkeypatch.setattr(retrieve_mod, "rerank", fake_rerank)
    return calls


@pytest.fixture
def config(monkeypatch):
    """Set retrieval settings for one test without touching the real .env."""

    def set(**kwargs):
        for name, value in kwargs.items():
            monkeypatch.setattr(retrieve_mod.settings, name, value)

    return set


# the candidate pool: too narrow a pool quietly makes reranking pointless


def test_retrieve_widens_the_pool_to_candidates_when_reranking(pipeline, config):
    """The reranker can only reorder what it is given: it must see `candidates`
    pages, not the k we ultimately want."""
    config(rerank=True, candidates=30, retrieval_mode="hybrid")

    retrieve_mod.retrieve("q", k=10)

    assert pipeline["hybrid"][0]["k"] == 30
    assert pipeline["rerank"][0]["k"] == 10
    assert len(pipeline["rerank"][0]["points"]) == 30


def test_retrieve_fetches_only_k_when_reranking_is_off(pipeline, config):
    config(rerank=False, candidates=30, retrieval_mode="hybrid")

    retrieve_mod.retrieve("q", k=10)

    assert pipeline["hybrid"][0]["k"] == 10
    assert pipeline["rerank"] == []


def test_retrieve_returns_exactly_k_pages_after_reranking(pipeline, config):
    config(rerank=True, candidates=30, retrieval_mode="hybrid")

    points = retrieve_mod.retrieve("q", k=10)

    assert len(points) == 10


# mode dispatch


def test_retrieve_uses_hybrid_search_in_hybrid_mode(pipeline, config):
    config(retrieval_mode="hybrid", rerank=False)

    retrieve_mod.retrieve("what were total assets?", k=5)

    assert pipeline["hybrid"][0]["query"] == "what were total assets?"
    assert pipeline["dense"] == []


def test_retrieve_uses_dense_search_in_dense_mode(pipeline, config):
    config(retrieval_mode="dense", rerank=False)

    retrieve_mod.retrieve("what were total assets?", k=5)

    assert pipeline["dense"][0]["query"] == "what were total assets?"
    assert pipeline["hybrid"] == []


def test_retrieve_reranks_dense_candidates_too(pipeline, config):
    config(retrieval_mode="dense", rerank=True, candidates=25)

    retrieve_mod.retrieve("q", k=5)

    assert pipeline["dense"][0]["k"] == 25
    assert len(pipeline["rerank"][0]["points"]) == 25
