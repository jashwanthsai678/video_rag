"""Pure retrieval logic: keyword scoring, rank fusion, and temporal clustering.

Every decision that determines *which* moment gets returned lives here, and
this module imports nothing heavier than the standard library. That is
deliberate: the ranking logic is the part most likely to be wrong, so it has
to be unit-testable without downloading a single model weight.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, deliberately conservative stopword list. Screen-recording text is
# mostly UI labels, so we only strip words that carry no retrieval signal at
# all — question scaffolding ("how do I ...") and articles.
_STOPWORDS = frozenset(
    """
    a an and are as at be by can do does did for from had has have how i in is it
    its me my of on or that the their there these this to was were what when
    where which who why will with you your
    """.split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


# --- Keyword retrieval -------------------------------------------------------


class BM25:
    """Okapi BM25 over pre-tokenized documents.

    Screen text is full of exact-match tokens (button labels, menu names,
    field captions) that dense embeddings blur together, so keyword scoring
    recovers hits pure vector search misses. Backed by an inverted index, so
    a query only scores documents that actually contain one of its terms.
    """

    def __init__(self, corpus: Iterable[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        for doc_id, tokens in enumerate(corpus):
            self.doc_lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term].append((doc_id, freq))

        n = len(self.doc_lengths)
        self.num_docs = n
        self.avg_doc_length = (sum(self.doc_lengths) / n) if n else 0.0
        self.idf = {
            term: math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> Dict[int, float]:
        """Sparse BM25 scores: only documents matching >=1 query term."""
        if not self.num_docs or not query_tokens:
            return {}

        out: Dict[int, float] = defaultdict(float)
        avgdl = self.avg_doc_length or 1.0
        for term in set(query_tokens):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for doc_id, freq in plist:
                norm = 1.0 - self.b + self.b * (self.doc_lengths[doc_id] / avgdl)
                out[doc_id] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)
        return dict(out)

    def top_n(self, query_tokens: Sequence[str], n: int) -> List[Tuple[int, float]]:
        ranked = sorted(self.scores(query_tokens).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:n]


# --- Rank fusion -------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> Dict[int, float]:
    """Reciprocal Rank Fusion over several ranked ID lists (best first).

    RRF combines rankings rather than raw scores, so cosine similarity and
    BM25 — which live on completely different scales — can be merged without
    inventing a normalization scheme that breaks on the next corpus.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must be the same length as rankings")

    fused: Dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, doc_id in enumerate(ranking):
            fused[int(doc_id)] += weight / (k + rank + 1)
    return dict(fused)


# --- Aggregation -------------------------------------------------------------


def rank_score(scores: Iterable[float], top_n: int = 3) -> float:
    """Sum of the best `top_n` scores — used to *choose* between candidates.

    A mean is the wrong aggregator here: it penalizes a video for also
    matching in weaker places, so a video with one mediocre hit can beat a
    video with three strong ones. Summing the top few rewards concentrated
    evidence without letting a long tail of weak hits dominate.
    """
    ranked = sorted(scores, reverse=True)[: max(1, top_n)]
    return float(sum(ranked))


def confidence(scores: Iterable[float]) -> float:
    """Strength of the single best piece of evidence.

    Kept separate from `rank_score` because thresholding needs a value on a
    stable, interpretable scale (cosine similarity), not a sum whose range
    depends on how many results came back.
    """
    values = list(scores)
    return float(max(values)) if values else 0.0


# --- Temporal clustering -----------------------------------------------------


@dataclass(frozen=True)
class TimeSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def cluster_by_time(
    spans: Sequence[Tuple[float, float]], max_gap_seconds: float
) -> List[List[int]]:
    """Group spans into runs that are contiguous in time.

    Without this, a query matching at 00:12 and again at 14:30 produces a
    fourteen-minute "clip" — technically containing the answer, useless as one.
    Returns lists of indices into `spans`, ordered by time, each with no
    internal gap larger than `max_gap_seconds`.
    """
    if not spans:
        return []

    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], spans[i][1]))
    groups: List[List[int]] = []
    current: List[int] = []
    current_end = 0.0

    for i in order:
        start, end = spans[i]
        end = max(start, end)
        if current and start - current_end > max_gap_seconds:
            groups.append(current)
            current = []
        current.append(i)
        current_end = end if len(current) == 1 else max(current_end, end)

    if current:
        groups.append(current)
    return groups


def span_of(spans: Sequence[Tuple[float, float]], indices: Iterable[int]) -> TimeSpan:
    """Bounding time span of the selected entries."""
    chosen = [spans[i] for i in indices]
    if not chosen:
        return TimeSpan(0.0, 0.0)
    return TimeSpan(min(s for s, _ in chosen), max(max(s, e) for s, e in chosen))


def group_by(items: Iterable, key) -> Dict:
    """Stable grouping helper: key -> list of items, insertion-ordered."""
    out: Dict = {}
    for item in items:
        out.setdefault(key(item), []).append(item)
    return out
