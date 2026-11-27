from .retrieve import Ranked, bm25_terms


def explain_miss(question: str, gold: str, ranked: Ranked, rank: int | None) -> dict:
    if rank is None:
        return {"reason": "答案被切断"}

    question_terms = set(bm25_terms(question))
    gold_terms = set(bm25_terms(gold))
    if question_terms.isdisjoint(gold_terms):
        return {
            "reason": "关键词不重合",
            "question_terms": sorted(question_terms),
            "gold_terms": sorted(gold_terms),
        }

    return {"reason": "被相似段落挤掉", "ahead": [chunk.id for chunk, _ in ranked[: rank - 1]]}
