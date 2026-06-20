from fastembed.rerank.cross_encoder import TextCrossEncoder
from .config import RERANK_MODEL

# Lazy initialization
_reranker = None


def _model():
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def rerank(query, points, k):
    """Rerank retrieved points with a cross-encoder over pairs and return the top k by reranker score."""
    if not points:
        return points
    scores = _model().rerank(query, [p.payload["content"] for p in points])
    ranked = sorted(zip(scores, points), key=lambda sp: sp[0], reverse=True)
    return [point for _, point in ranked[:k]]
