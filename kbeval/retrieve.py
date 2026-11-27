from collections import Counter
from math import log1p
import re

from .chunk import Chunk


_CHINESE = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af"
_RUNS = re.compile(rf"[{_CHINESE}]+|[a-z0-9]+")
Ranked = list[tuple[Chunk, float]]


def _bigrams(run: str):
    return (run[i : i + 2] for i in range(len(run) - 1))


def bm25_terms(text: str) -> list[str]:
    terms = []
    for run in _RUNS.findall(text.lower()):
        if run.isascii():
            terms.append(run)
        else:
            terms.extend(run)
            terms.extend(_bigrams(run))
    return terms


def _rank(scored: Ranked) -> Ranked:
    return sorted(scored, key=lambda item: (-item[1], item[0].document, item[0].start))


class BM25:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.counts = [Counter(bm25_terms(chunk.text)) for chunk in chunks]
        lengths = [sum(counts.values()) for counts in self.counts]
        average_length = (sum(lengths) / len(lengths) if lengths else 0) or 1
        self.k1 = 1.2
        b = 0.75
        self.length_norms = [self.k1 * (1 - b + b * length / average_length) for length in lengths]
        frequencies = Counter(term for counts in self.counts for term in counts)
        self.idf = {
            term: log1p((len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in frequencies.items()
        }

    def rank(self, question: str) -> Ranked:
        terms = sorted(set(bm25_terms(question)) & self.idf.keys())
        scored = []
        for chunk, counts, length_norm in zip(self.chunks, self.counts, self.length_norms):
            score = sum(
                self.idf[term] * counts[term] * (self.k1 + 1) / (counts[term] + length_norm)
                for term in terms
                if counts[term]
            )
            scored.append((chunk, score))
        return _rank(scored)


def _bigram_set(text: str) -> set[str]:
    return {pair for run in _RUNS.findall(text.lower()) for pair in _bigrams(run)}


class BigramOverlap:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.terms = [_bigram_set(chunk.text) for chunk in chunks]

    def rank(self, question: str) -> Ranked:
        query = _bigram_set(question)
        return _rank([(chunk, len(query & terms)) for chunk, terms in zip(self.chunks, self.terms)])
