def reciprocal_rank(rank: int | None, k: int | None = None) -> float:
    return 1 / rank if rank is not None and (k is None or rank <= k) else 0.0


def summarize(ranks: dict[str, int | None], k: int) -> dict:
    hits = {qid: rank for qid, rank in ranks.items() if rank is not None and rank <= k}
    return {
        "recall": len(hits) / len(ranks),
        "mrr": sum(reciprocal_rank(rank) for rank in ranks.values()) / len(ranks),
        "late_hits": {qid: rank for qid, rank in hits.items() if rank > 1},
    }
