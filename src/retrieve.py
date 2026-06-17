from .embed import embed_texts
from .store import client
from .config import QDRANT_COLLECTION


def retrieve(query, k=5):
    """Retrieve the top k most similar documents to the query from the Qdrant collection."""
    query_vec = embed_texts([query])[0]
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vec,
        limit=k,
        with_payload=True,
    )
    return response.points
