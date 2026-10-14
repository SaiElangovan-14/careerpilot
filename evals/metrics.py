def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def hit_rate_at_k(retrieved: list[int], relevant :set[int], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0

def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for rank, item_id in enumerate(retrieved, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0