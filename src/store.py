from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
)
from .config import settings

client = QdrantClient(url=settings.qdrant_url)

# Named vectors. The dense vector holds the embedding of each page's LLM
# summary; the sparse vector is BM25 over the full page text.
DENSE = "summary"
SPARSE = "sparse"


def init_collection():
    """Initialize the Qdrant collection with named dense and sparse vectors. Recreates if it exists."""
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={
            DENSE: VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            SPARSE: SparseVectorParams(),
        },
    )


def upsert_pages(pages, dense_vectors, sparse_vectors, start_id):
    """Upsert page records with their dense and sparse vectors into the Qdrant collection."""
    points = [
        PointStruct(
            id=start_id + i,
            vector={
                DENSE: dense,
                SPARSE: SparseVector(
                    indices=sparse.indices.tolist(),
                    values=sparse.values.tolist(),
                ),
            },
            payload={
                "company": p["company"],
                "doc_name": p["doc_name"],
                "page": p["page"],
                "content": p["content"],
                "summary": p["summary"],
            },
        )
        for i, (p, dense, sparse) in enumerate(
            zip(pages, dense_vectors, sparse_vectors)
        )
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
