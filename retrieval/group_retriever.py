"""
retrieve_for_group — выполняет RAG-запросы для всех полей одной группы.

Поддерживает два режима:

  "per_field" (по умолчанию):
      Для каждого поля — отдельный dual_retrieve (Text + TableRow).
      Каждое поле получает свой набор чанков.

  "group_deduplicated":
      Те же N индивидуальных dual_retrieve, но результаты объединяются,
      дедуплицируются по тексту и обрезаются до group_k чанков по score.
      Все поля группы получают один общий пул — меньше дублирования в промпте.

Возвращает:
  - field_contexts: dict[field_id, list[RetrievedChunk]]
  - retrieval_logs: list[FieldRetrievalLog]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from config.field_config import FIELD_CONFIG
from retrieval.retriever import RetrievedChunk, dual_retrieve
from run_log.run_logger import FieldRetrievalLog

if TYPE_CHECKING:
    from retrieval.store import VectorStoreRegistry

RetrievalMode = Literal["per_field", "group_deduplicated"]


def retrieve_for_group(
    registry: "VectorStoreRegistry",
    field_ids: list[str],
    k_text: int = 10,
    k_table: int = 6,
    retrieval_mode: RetrievalMode = "per_field",
    group_k: int = 20,
) -> tuple[dict[str, list[RetrievedChunk]], list[FieldRetrievalLog]]:
    """
    RAG-ретрив для всех полей группы.

    Args:
        registry:        Зарегистрированные векторные базы.
        field_ids:       Список ID полей в группе.
        k_text:          Чанков из Text-слоя на каждый запрос.
        k_table:         Чанков из TableRow-слоя на каждый запрос.
        retrieval_mode:  "per_field" — каждое поле получает свои чанки.
                         "group_deduplicated" — общий дедуплицированный пул.
        group_k:         Итоговый размер пула в режиме group_deduplicated.

    Returns:
        (field_contexts, retrieval_logs)
    """
    # --- Шаг 1: индивидуальный dual_retrieve для каждого поля ---
    per_field_chunks: dict[str, list[RetrievedChunk]] = {}
    retrieval_logs: list[FieldRetrievalLog] = []

    for field_id in field_ids:
        cfg = FIELD_CONFIG[field_id]
        collection_name: str = cfg.get("collection", "default")
        rag_query: str = cfg["rag_query"]

        chunks = dual_retrieve(
            registry=registry,
            collection_name=collection_name,
            query=rag_query,
            k_text=k_text,
            k_table=k_table,
        )

        per_field_chunks[field_id] = chunks

        retrieval_logs.append(
            FieldRetrievalLog(
                field_id=field_id,
                rag_query=rag_query,
                retrieved_chunks=[
                    {
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "score": chunk.score,
                        "source": chunk.source,
                    }
                    for chunk in chunks
                ],
            )
        )

    # --- Шаг 2: сборка field_contexts по режиму ---
    if retrieval_mode == "per_field":
        field_contexts = per_field_chunks

    else:  # group_deduplicated
        # Объединяем все чанки, дедуплицируем по тексту, сортируем по score
        seen: set[str] = set()
        all_chunks: list[RetrievedChunk] = []

        for chunks in per_field_chunks.values():
            for chunk in chunks:
                if chunk.text not in seen:
                    seen.add(chunk.text)
                    all_chunks.append(chunk)

        shared = sorted(all_chunks, key=lambda c: c.score, reverse=True)[:group_k]
        field_contexts = {fid: shared for fid in field_ids}

    return field_contexts, retrieval_logs
