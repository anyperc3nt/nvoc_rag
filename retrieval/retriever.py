"""
hybrid_retrieve — гибридный поиск: векторный (ChromaDB) + BM25 + Reciprocal Rank Fusion.

Если BM25-индекс для коллекции не был зарегистрирован (corpus_path не передавался),
возвращается только результат векторного поиска.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from retrieval.store import VectorStoreRegistry


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float          # итоговый RRF-score (чем выше — тем релевантнее)
    source: str           # "vector", "bm25", или "rrf"


def _rrf_merge(
    vector_hits: list[tuple[str, dict]],   # [(text, metadata), ...]
    bm25_hits: list[tuple[str, dict]],
    k_rrf: int = 60,
) -> list[tuple[str, dict, float]]:
    """
    Reciprocal Rank Fusion.

    Формула: score(d) = sum( 1 / (k + rank_i) )
    Ранги считаются раздельно для каждого списка (1-based).
    """
    scores: dict[str, float] = {}
    meta_map: dict[str, dict] = {}

    for rank, (text, meta) in enumerate(vector_hits, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k_rrf + rank)
        meta_map[text] = meta

    for rank, (text, meta) in enumerate(bm25_hits, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (k_rrf + rank)
        meta_map.setdefault(text, meta)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(text, meta_map[text], score) for text, score in ranked]


def hybrid_retrieve(
    registry: "VectorStoreRegistry",
    collection_name: str,
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    filters: Optional[dict] = None,
) -> list[RetrievedChunk]:
    """
    Гибридный поиск по одному запросу.

    Args:
        registry:         VectorStoreRegistry с зарегистрированными коллекциями.
        collection_name:  Имя коллекции (из FIELD_CONFIG["collection"]).
        query:            Текстовый запрос.
        k:                Количество результатов из каждого метода поиска.
        alpha:            Не используется напрямую (RRF сам балансирует); оставлен
                          для будущей параметризации весов.
        filters:          Метадата-фильтры для ChromaDB (например, {"doc_type": "..."}).

    Returns:
        Список RetrievedChunk, отсортированный по убыванию RRF-score,
        усечённый до k элементов.
    """
    # --- 1. Векторный поиск ---
    vector_results = registry.vector_query(collection_name, query, k=k, filters=filters)
    vector_hits: list[tuple[str, dict]] = [
        (doc.page_content, doc.metadata) for doc, _score in vector_results
    ]

    # --- 2. BM25 ---
    bm25_index = registry.get_bm25(collection_name)
    bm25_hits: list[tuple[str, dict]] = []

    if bm25_index is not None:
        corpus = registry.get_bm25_corpus(collection_name)
        tokenized_query = query.split()
        bm25_scores = bm25_index.get_scores(tokenized_query)

        # Берём top-k по BM25 (индексы отсортированы по убыванию score)
        top_k_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:k]

        bm25_hits = [(corpus[i], {}) for i in top_k_indices if bm25_scores[i] > 0]

    # --- 3. Если BM25 недоступен — возвращаем только вектор ---
    if not bm25_hits:
        return [
            RetrievedChunk(
                text=text,
                metadata=meta,
                score=float(k - idx),   # простой убывающий score по рангу
                source="vector",
            )
            for idx, (text, meta) in enumerate(vector_hits[:k])
        ]

    # --- 4. Reciprocal Rank Fusion ---
    merged = _rrf_merge(vector_hits, bm25_hits)[:k]

    return [
        RetrievedChunk(text=text, metadata=meta, score=score, source="rrf")
        for text, meta, score in merged
    ]
