from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
)
from .config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM

client = QdrantClient(url=QDRANT_URL)

# Named vector identifiers used across ingest and retrieval
DENSE = "dense"
SPARSE = "sparse"


def init_collection():
    """Initialize the Qdrant collection with named dense and sparse vectors. Recreates if it exists."""
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            DENSE: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
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
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
