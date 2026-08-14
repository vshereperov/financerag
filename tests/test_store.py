import numpy as np
import pytest

from src import store


class FakeQdrant:
    """Stands in for the Qdrant client so nothing is written to a real collection."""

    def __init__(self):
        self.upserts = []

    def upsert(self, collection_name, points):
        self.upserts.append({"collection_name": collection_name, "points": points})


@pytest.fixture
def qdrant(monkeypatch):
    fake = FakeQdrant()
    monkeypatch.setattr(store, "client", fake)
    return fake


def page(page_num, company="AMD", doc_name="AMD_2022_10K"):
    return {
        "company": company,
        "doc_name": doc_name,
        "page": page_num,
        "content": f"full text of page {page_num}",
        "summary": f"summary of page {page_num}",
    }


def sparse(indices=(1, 7), values=(0.5, 0.25)):
    """fastembed hands back numpy arrays, which store.py converts with .tolist()."""
    return type(
        "SparseEmbedding",
        (),
        {"indices": np.array(indices), "values": np.array(values)},
    )()


def upserted_points(qdrant):
    return qdrant.upserts[0]["points"]


# point ids: a collision here silently overwrites pages already ingested


def test_upsert_numbers_points_from_start_id(qdrant):
    """ingest() writes in batches of 100 and passes the running offset. If ids
    restarted at 0 each batch, every batch would overwrite the previous one."""
    pages = [page(1), page(2), page(3)]

    store.upsert_pages(pages, [[0.1]] * 3, [sparse()] * 3, start_id=100)

    assert [p.id for p in upserted_points(qdrant)] == [100, 101, 102]


def test_upsert_ids_do_not_collide_across_batches(qdrant):
    store.upsert_pages([page(1), page(2)], [[0.1]] * 2, [sparse()] * 2, start_id=0)
    store.upsert_pages([page(3), page(4)], [[0.1]] * 2, [sparse()] * 2, start_id=2)

    ids = [p.id for upsert in qdrant.upserts for p in upsert["points"]]
    assert ids == [0, 1, 2, 3]
    assert len(set(ids)) == 4


# payload and vectors


def test_upsert_stores_the_fields_retrieval_and_the_eval_read_back(qdrant):
    store.upsert_pages([page(42)], [[0.1, 0.2]], [sparse()], start_id=0)

    payload = upserted_points(qdrant)[0].payload
    assert payload == {
        "company": "AMD",
        "doc_name": "AMD_2022_10K",
        "page": 42,
        "content": "full text of page 42",
        "summary": "summary of page 42",
    }


def test_upsert_pairs_each_page_with_its_own_vectors(qdrant):
    """The three lists are zipped: page N must not get page M's embedding."""
    pages = [page(1), page(2)]
    dense = [[1.0, 1.0], [2.0, 2.0]]
    sparses = [sparse([1], [0.9]), sparse([7], [0.4])]

    store.upsert_pages(pages, dense, sparses, start_id=0)

    first, second = upserted_points(qdrant)
    assert first.vector[store.DENSE] == [1.0, 1.0]
    assert first.vector[store.SPARSE].indices == [1]
    assert second.vector[store.DENSE] == [2.0, 2.0]
    assert second.vector[store.SPARSE].indices == [7]


def test_upsert_converts_numpy_sparse_vectors_to_plain_lists(qdrant):
    """Qdrant cannot serialize numpy arrays, so .tolist() has to happen here."""
    store.upsert_pages([page(1)], [[0.1]], [sparse((3, 9), (0.75, 0.25))], start_id=0)

    vector = upserted_points(qdrant)[0].vector[store.SPARSE]
    assert vector.indices == [3, 9]
    assert vector.values == [0.75, 0.25]


def test_upsert_writes_to_the_configured_collection(qdrant):
    store.upsert_pages([page(1)], [[0.1]], [sparse()], start_id=0)

    assert qdrant.upserts[0]["collection_name"] == store.settings.qdrant_collection
