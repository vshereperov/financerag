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

# Named vector identifiers used across ingest and retrieval
DENSE = "dense"
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


def upsert_chunks(chunks, dense_vectors, sparse_vectors, start_id):
    """Upsert chunks with their dense and sparse vectors into the Qdrant collection."""
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
                "company": c["company"],
                "doc_name": c["doc_name"],
                "page": c["page"],
                "content": c["content"],
            },
        )
        for i, (c, dense, sparse) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        )
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
