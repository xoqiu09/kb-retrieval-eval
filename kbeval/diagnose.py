from .retrieve import Ranked, bigram_set, bm25_terms


def explain_miss(question: str, gold: str, ranked: Ranked, rank: int | None, k: int, *, retriever: str = "bm25") -> dict:
    if rank is None:
        return {"reason": "答案被切断"}

    ahead = [chunk.id for chunk, _ in ranked[: rank - 1]]
    score = ranked[rank - 1][1]
    if score == ranked[k - 1][1]:
        return {"reason": "同分，按文件名和块起点排序", "tied_score": score, "ahead": ahead}

    terms = bigram_set if retriever == "bigram" else bm25_terms
    question_terms = set(terms(question))
    gold_terms = set(terms(gold))
    if question_terms.isdisjoint(gold_terms):
        return {
            "reason": "关键词不重合",
            "question_terms": sorted(question_terms),
            "gold_terms": sorted(gold_terms),
        }

    return {"reason": "被相似段落挤掉", "ahead": ahead}
