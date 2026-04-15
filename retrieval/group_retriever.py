"""
retrieve_for_group — выполняет RAG-запросы для всех полей одной группы.

Возвращает:
  - field_contexts: dict[field_id, list[RetrievedChunk]]  — чанки по каждому полю
  - retrieval_logs: list[FieldRetrievalLog]               — для логгера
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.field_config import FIELD_CONFIG
from retrieval.retriever import RetrievedChunk, hybrid_retrieve
from run_log.run_logger import FieldRetrievalLog

if TYPE_CHECKING:
    from retrieval.store import VectorStoreRegistry


def retrieve_for_group(
    registry: "VectorStoreRegistry",
    field_ids: list[str],
    k: int = 5,
) -> tuple[dict[str, list[RetrievedChunk]], list[FieldRetrievalLog]]:
    """
    Для каждого поля из группы выполняет hybrid_retrieve и собирает логи.

    Args:
        registry:    Зарегистрированные векторные базы.
        field_ids:   Список ID полей (из активной стратегии группировки).
        k:           Количество чанков на каждый RAG-запрос.

    Returns:
        (field_contexts, retrieval_logs)
        field_contexts  — dict[field_id -> [RetrievedChunk, ...]]
        retrieval_logs  — список FieldRetrievalLog для RunLogger
    """
    field_contexts: dict[str, list[RetrievedChunk]] = {}
    retrieval_logs: list[FieldRetrievalLog] = []

    for field_id in field_ids:
        cfg = FIELD_CONFIG[field_id]
        collection_name: str = cfg.get("collection", "default")
        rag_query: str = cfg["rag_query"]

        chunks = hybrid_retrieve(
            registry=registry,
            collection_name=collection_name,
            query=rag_query,
            k=k,
        )

        field_contexts[field_id] = chunks

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

    return field_contexts, retrieval_logs
