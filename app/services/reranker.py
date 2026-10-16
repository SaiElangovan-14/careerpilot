from sentence_transformers import CrossEncoder

from app.services.retrieval import hybrid_search

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def search_with_rerank(session, query: str, top_k: int = 5, candidates: int = 20):
    found = hybrid_search(session, query, top_k=candidates, mode="hybrid")
    if not found:
        return []

    jobs = [job for job, _score in found]
    pairs = [(query, f"{job.title} {job.description}") for job in jobs]
    scores = get_reranker().predict(pairs)

    ranked = sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True)
    return [(job, float(score)) for job, score in ranked[:top_k]]
