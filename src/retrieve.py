from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
from .embed import embed_texts, embed_sparse_query
from .store import client, DENSE, SPARSE
from .rerank import rerank
from .config import settings


def _retrieve_dense(query, k):
    """Dense retrieval: retrieve the top k candidates from the dense vector index."""
    dense_vec = embed_texts([query])[0]
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        query=dense_vec,
        using=DENSE,
        limit=k,
        with_payload=True,
    )
    return response.points


def _retrieve_hybrid(query, k):
    """Hybrid retrieval: fuse dense and sparse candidates with RRF."""
    dense_vec = embed_texts([query])[0]
    sparse_vec = embed_sparse_query(query)
    response = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE, limit=settings.candidates),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using=SPARSE,
                limit=settings.candidates,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return response.points


def retrieve(query, k):
    """Retrieve the top k pages for a query, dispatching on RETRIEVAL_MODE.

    When RERANK is enabled, retrieve a wider candidate pool and re-score it
    down to k with the hosted rerank API.
    """
    fetch_k = settings.candidates if settings.rerank else k
    if settings.retrieval_mode == "dense":
        points = _retrieve_dense(query, fetch_k)
    else:
        points = _retrieve_hybrid(query, fetch_k)
    if settings.rerank:
        points = rerank(query, points, k)
    return points
