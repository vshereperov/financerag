from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from .config import QDRANT_URL, QDRANT_COLLECTION, EMBEDDING_DIM

client = QdrantClient(url=QDRANT_URL)


def init_collection():
    """Initialize the Qdrant collection. If it already exists, delete and recreate it."""
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def upsert_chunks(chunks, vectors, start_id):
    """Upsert chunks and their corresponding vectors into the Qdrant collection."""
    points = [
        PointStruct(
            id=start_id + i,
            vector=vec,
            payload={
                "company": c["company"],
                "doc_name": c["doc_name"],
                "page": c["page"],
                "content": c["content"],
            },
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
