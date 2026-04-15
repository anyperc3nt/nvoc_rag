"""
run_pipeline — оркестратор пайплайна НВОС.

Для каждой группы из активной стратегии:
  1. RAG-ретрив по каждому полю (hybrid_retrieve)
  2. Сборка промпта для всей группы
  3. Вызов LLM (extract_group)
  4. Логирование результатов
  5. Сборка итогового словаря

Результат: dict[field_id -> Pydantic-объект | None]
"""

import time

import instructor

from config.field_config import FIELD_CONFIG
from config.group_config import get_active_groups, get_grouping_meta
from extraction.extractor import extract_group, parsed_to_loggable
from extraction.prompts import SYSTEM_PROMPT, build_group_prompt
from extraction.schemas import NotFound
from retrieval.group_retriever import retrieve_for_group
from retrieval.store import VectorStoreRegistry
from run_log.run_logger import GroupCallLog, RunLogger


def run_pipeline(
    registry: VectorStoreRegistry,
    llm_client: instructor.Instructor,
    model_name: str,
    logger: RunLogger,
    retrieval_k: int = 5,
) -> dict:
    """
    Запускает полный пайплайн заполнения заявки НВОС.

    Args:
        registry:     Зарегистрированные векторные базы.
        llm_client:   Instructor-клиент (vLLM или OpenAI).
        model_name:   Имя модели для вызова LLM.
        logger:       RunLogger для записи хода выполнения.
        retrieval_k:  Количество чанков на каждый RAG-запрос.

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
            k=retrieval_k,
        )

        # --- 2. Сборка промпта ---
        user_prompt = build_group_prompt(field_ids, field_contexts)

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
