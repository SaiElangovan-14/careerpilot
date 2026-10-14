_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer 

        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vectors]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x,y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)