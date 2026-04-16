"""
Функции поиска по векторной базе.

hybrid_retrieve — классический гибрид: вектор + BM25 + RRF по одному фильтру.
dual_retrieve   — параллельный поиск по двум срезам базы:
                    Text (k_text чанков) + TableRow (k_table чанков).
                  AllTableMD исключается жёстко: он не передаётся ни в один фильтр.
                  BM25 применяется только к Text-результатам (если индекс доступен).
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
        tokenized_query = query.lower().split()
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


def dual_retrieve(
    registry: "VectorStoreRegistry",
    collection_name: str,
    query: str,
    k_text: int = 10,
    k_table: int = 6,
) -> list[RetrievedChunk]:
    """
    Параллельный поиск по двум срезам базы: Text и TableRow.

    AllTableMD исключается жёстко — слишком длинные фрагменты (~4k символов)
    переполняют контекст LLM.

    BM25 применяется только к Text-результатам: индекс строится по Text-чанкам
    (bm25_record_types=["Text"] в store.register). Для TableRow BM25 не используется,
    т.к. строки таблиц обычно короткие и хорошо находятся вектором.

    Args:
        registry:         VectorStoreRegistry.
        collection_name:  Имя коллекции.
        query:            Текстовый запрос.
        k_text:           Чанков из Text-слоя.
        k_table:          Чанков из TableRow-слоя.

    Returns:
        Объединённый список RetrievedChunk, отсортированный по score (Text RRF + TableRow вектор).
    """
    # --- 1. Векторный поиск по Text ---
    text_vector = registry.vector_query(
        collection_name, query, k=k_text,
        filters={"record_type": "Text"},
    )
    text_vector_hits: list[tuple[str, dict]] = [
        (doc.page_content, doc.metadata) for doc, _ in text_vector
    ]

    # --- 2. BM25 по Text (если индекс доступен) ---
    bm25_index = registry.get_bm25(collection_name)
    bm25_hits: list[tuple[str, dict]] = []
    if bm25_index is not None:
        corpus = registry.get_bm25_corpus(collection_name)
        bm25_scores = bm25_index.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:k_text]
        bm25_hits = [(corpus[i], {}) for i in top_indices if bm25_scores[i] > 0]

    # --- 3. RRF по Text ---
    if bm25_hits:
        text_merged = _rrf_merge(text_vector_hits, bm25_hits)[:k_text]
        text_chunks = [
            RetrievedChunk(text=t, metadata=m, score=s, source="rrf")
            for t, m, s in text_merged
        ]
    else:
        text_chunks = [
            RetrievedChunk(text=t, metadata=m, score=float(k_text - i), source="vector")
            for i, (t, m) in enumerate(text_vector_hits[:k_text])
        ]

    # --- 4. Векторный поиск по TableRow ---
    table_vector = registry.vector_query(
        collection_name, query, k=k_table,
        filters={"record_type": "TableRow"},
    )
    table_chunks = [
        RetrievedChunk(
            text=doc.page_content,
            metadata=doc.metadata,
            score=float(k_table - i),
            source="vector",
        )
        for i, (doc, _) in enumerate(table_vector)
    ]

    # --- 5. Объединяем Text + TableRow ---
    return text_chunks + table_chunks
