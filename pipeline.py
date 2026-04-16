"""
run_pipeline — оркестратор пайплайна НВОС.

Для каждой группы из активной стратегии:
  1. RAG-ретрив (dual_retrieve: Text + TableRow) в выбранном режиме
  2. Сборка промпта для всей группы
  3. Вызов LLM (extract_group)
  4. Логирование результатов
  5. Сборка итогового словаря

Результат: dict[field_id -> Pydantic-объект | None]
"""

import time
from typing import Literal

import instructor

from config.group_config import get_active_groups, get_grouping_meta
from extraction.extractor import extract_group, parsed_to_loggable
from extraction.prompts import (
    SYSTEM_PROMPT,
    build_per_field_prompt,
    build_shared_context_prompt,
)
from extraction.schemas import NotFound
from retrieval.group_retriever import retrieve_for_group
from retrieval.store import VectorStoreRegistry
from run_log.run_logger import GroupCallLog, RunLogger

RetrievalMode = Literal["per_field", "group_deduplicated"]


def run_pipeline(
    registry: VectorStoreRegistry,
    llm_client: instructor.Instructor,
    model_name: str,
    logger: RunLogger,
    retrieval_mode: RetrievalMode = "per_field",
    k_text: int = 10,
    k_table: int = 6,
    group_k: int = 20,
) -> dict:
    """
    Запускает полный пайплайн заполнения заявки НВОС.

    Args:
        registry:        Зарегистрированные векторные базы.
        llm_client:      Instructor-клиент (vLLM или OpenAI).
        model_name:      Имя модели для вызова LLM.
        logger:          RunLogger для записи хода выполнения.
        retrieval_mode:  "per_field" — каждое поле получает свои чанки.
                         "group_deduplicated" — общий дедуплицированный пул на группу.
        k_text:          Чанков из Text-слоя на каждый RAG-запрос.
        k_table:         Чанков из TableRow-слоя на каждый RAG-запрос.
        group_k:         Размер итогового пула в режиме group_deduplicated.

    Returns:
        dict[field_id -> Pydantic-объект | None]
        None означает, что поле не найдено (NotFound или ошибка).
    """
    active_groups = get_active_groups()
    grouping_meta = get_grouping_meta()

    print(
        f"[Pipeline] Стратегия: '{grouping_meta['strategy_name']}', "
        f"групп: {grouping_meta['num_groups']}, "
        f"полей: {sum(g['num_fields'] for g in grouping_meta['groups'].values())}"
    )

    result: dict = {}

    for group_id, group_cfg in active_groups.items():
        field_ids: list[str] = group_cfg["fields"]
        group_name: str = group_cfg["name"]

        print(f"[Pipeline] → Группа '{group_id}' ({group_name}): {len(field_ids)} полей")
        t0 = time.time()

        # --- 1. RAG-ретрив ---
        field_contexts, retrieval_logs = retrieve_for_group(
            registry=registry,
            field_ids=field_ids,
            k_text=k_text,
            k_table=k_table,
            retrieval_mode=retrieval_mode,
            group_k=group_k,
        )

        # --- 2. Сборка промпта ---
        if retrieval_mode == "group_deduplicated":
            shared_chunks = field_contexts[field_ids[0]]
            user_prompt = build_shared_context_prompt(field_ids, shared_chunks)
        else:
            user_prompt = build_per_field_prompt(field_ids, field_contexts)

        # --- 3. Вызов LLM ---
        try:
            llm_response_raw, parsed = extract_group(
                llm_client=llm_client,
                model_name=model_name,
                field_ids=field_ids,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            print(f"[Pipeline] ОШИБКА в группе '{group_id}': {exc}")
            llm_response_raw = f"ERROR: {exc}"
            parsed = {fid: None for fid in field_ids}

        # --- 4. Сбор результатов и issues ---
        for fid, value in parsed.items():
            if value is None or isinstance(value, NotFound):
                reason = value.reason if isinstance(value, NotFound) else "LLM вернул None"
                logger.log_issue(fid, reason)
                result[fid] = None
            else:
                result[fid] = value

        duration = time.time() - t0

        # --- 5. Логирование группы ---
        logger.log_group(
            GroupCallLog(
                group_id=group_id,
                group_name=group_name,
                fields_in_group=field_ids,
                retrieval_logs=retrieval_logs,
                system_prompt=SYSTEM_PROMPT,
                full_user_prompt=user_prompt,
                llm_response_raw=llm_response_raw,
                llm_response_parsed=parsed_to_loggable(parsed),
                duration_seconds=round(duration, 3),
            )
        )

        print(f"[Pipeline]   Готово за {duration:.1f}с")

    logger.finalize()
    return result
