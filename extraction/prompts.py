"""
Промпты для LLM-экстракции.

SYSTEM_PROMPT   — общий системный промпт для всех вызовов.
build_group_prompt() — формирует пользовательский промпт для группы полей.
"""

from __future__ import annotations

from retrieval.retriever import RetrievedChunk
from config.field_config import FIELD_CONFIG


SYSTEM_PROMPT = """\
Ты — эксперт по экологическому законодательству Российской Федерации.

Твоя задача — извлекать конкретные сведения из предоставленных фрагментов документов \
для заполнения заявки о постановке объекта НВОС на государственный учёт \
(форма по приказу Минприроды России № 532 от 12.08.2022).

Правила работы:
1. Отвечай строго на основе предоставленного контекста.
2. Если информация отсутствует, неоднозначна или не может быть однозначно \
   отнесена к запрашиваемому полю — возвращай NotFound с чётким объяснением причины.
3. Не додумывай, не интерполируй и не дополняй данные из общих знаний.
4. Для каждого заполненного поля указывай дословную цитату из контекста \
   (source_fragment) — это обязательно.
5. Оценивай уверенность (confidence) честно: 1.0 — данные прямо \
   присутствуют в тексте, 0.5 — выводимые данные, 0.0 — данных нет.
"""


def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Форматирует список чанков в читаемый блок контекста."""
    if not chunks:
        return "(контекст не найден)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta_str = ""
        if chunk.metadata:
            meta_parts = [f"{k}={v}" for k, v in chunk.metadata.items()]
            meta_str = f" [{', '.join(meta_parts)}]"
        parts.append(f"[Фрагмент {i}{meta_str}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def build_group_prompt(
    field_ids: list[str],
    field_contexts: dict[str, list[RetrievedChunk]],
) -> str:
    """
    Строит пользовательский промпт для группы полей.

    Структура промпта:
      - Для каждого поля: описание + notes + найденные фрагменты
      - Итоговая инструкция

    Args:
        field_ids:       Список ID полей в группе (сохраняет порядок).
        field_contexts:  dict[field_id -> [RetrievedChunk, ...]]

    Returns:
        Строка пользовательского промпта.
    """
    sections: list[str] = []

    for field_id in field_ids:
        cfg = FIELD_CONFIG[field_id]
        chunks = field_contexts.get(field_id, [])
        context_text = _format_chunks(chunks)

        notes_line = ""
        if cfg.get("notes"):
            notes_line = f"Пояснение: {cfg['notes']}\n"

        condition_line = ""
        if cfg.get("condition"):
            condition_line = f"Условие заполнения: {cfg['condition']}\n"

        section = (
            f"### {field_id}\n"
            f"Описание поля: {cfg['description']}\n"
            f"{notes_line}"
            f"{condition_line}"
            f"\nКонтекст из документов:\n{context_text}"
        )
        sections.append(section)

    fields_list = ", ".join(field_ids)

    prompt = (
        "Заполни следующие поля заявки НВОС на основе контекста из документов.\n\n"
        + "\n\n".join(sections)
        + f"\n\n---\n\nИзвлеки значения для полей: {fields_list}.\n"
        "Для каждого поля верни либо заполненную схему, либо NotFound с объяснением.\n"
        "Опирайся исключительно на приведённый выше контекст."
    )

    return prompt
