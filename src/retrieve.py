from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion
from .embed import embed_texts, embed_sparse_query
from .store import client, DENSE, SPARSE
from .rerank import rerank
from .config import QDRANT_COLLECTION, RETRIEVAL_MODE, RERANK

# Number of candidates to retrieve from each index for hybrid retrieval
CANDIDATES = 30

# Size of the candidate pool fed to the reranker when RERANK is enabled
RERANK_CANDIDATES = 30


def _retrieve_dense(query, k):
    """Dense retrieval: retrieve the top k candidates from the dense vector index."""
    dense_vec = embed_texts([query])[0]
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
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
        collection_name=QDRANT_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE, limit=CANDIDATES),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using=SPARSE,
                limit=CANDIDATES,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k,
        with_payload=True,
    )
    return response.points


def retrieve(query, k=5):
    """Retrieve the top k chunks for a query, dispatching on RETRIEVAL_MODE.

    When RERANK is enabled, retrieve a wider candidate pool and re-score it
    down to k with a cross-encoder.
    """
    fetch_k = RERANK_CANDIDATES if RERANK else k
    if RETRIEVAL_MODE == "dense":
        points = _retrieve_dense(query, fetch_k)
    else:
        points = _retrieve_hybrid(query, fetch_k)
    if RERANK:
        points = rerank(query, points, k)
    return points
