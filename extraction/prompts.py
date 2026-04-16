"""
Промпты для LLM-экстракции.

SYSTEM_PROMPT               — общий системный промпт для всех вызовов.
build_per_field_prompt()    — промпт для режима per_field: контекст у каждого поля свой.
build_shared_context_prompt() — промпт для режима group_deduplicated: контекст один раз,
                                затем список полей.
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


def _field_header(field_id: str) -> str:
    """Описание одного поля без контекста (для shared-режима)."""
    cfg = FIELD_CONFIG[field_id]
    lines = [f"### {field_id}", f"Описание: {cfg['description']}"]
    if cfg.get("notes"):
        lines.append(f"Пояснение: {cfg['notes']}")
    if cfg.get("condition"):
        lines.append(f"Условие заполнения: {cfg['condition']}")
    return "\n".join(lines)


def _field_section_with_context(field_id: str, chunks: list[RetrievedChunk]) -> str:
    """Описание поля вместе с его контекстом (для per_field-режима)."""
    cfg = FIELD_CONFIG[field_id]
    lines = [f"### {field_id}", f"Описание: {cfg['description']}"]
    if cfg.get("notes"):
        lines.append(f"Пояснение: {cfg['notes']}")
    if cfg.get("condition"):
        lines.append(f"Условие заполнения: {cfg['condition']}")
    lines.append(f"\nКонтекст из документов:\n{_format_chunks(chunks)}")
    return "\n".join(lines)


_PROMPT_FOOTER = (
    "Для каждого поля верни либо заполненную схему, либо NotFound с объяснением.\n"
    "Опирайся исключительно на приведённый выше контекст."
)


def build_per_field_prompt(
    field_ids: list[str],
    field_contexts: dict[str, list[RetrievedChunk]],
) -> str:
    """
    Промпт для режима per_field: каждое поле — со своим контекстом.

    Структура:
        [Описание + контекст поля 1]
        [Описание + контекст поля 2]
        ...
        Инструкция
    """
    sections = [
        _field_section_with_context(fid, field_contexts.get(fid, []))
        for fid in field_ids
    ]
    fields_list = ", ".join(field_ids)
    return (
        "Заполни следующие поля заявки НВОС на основе контекста из документов.\n\n"
        + "\n\n".join(sections)
        + f"\n\n---\n\nИзвлеки значения для полей: {fields_list}.\n"
        + _PROMPT_FOOTER
    )


def build_shared_context_prompt(
    field_ids: list[str],
    shared_chunks: list[RetrievedChunk],
) -> str:
    """
    Промпт для режима group_deduplicated: контекст выводится один раз,
    затем — список полей с описаниями.

    Структура:
        [Общий контекст — N чанков]
        [Описание поля 1]
        [Описание поля 2]
        ...
        Инструкция
    """
    context_block = _format_chunks(shared_chunks)
    field_headers = "\n\n".join(_field_header(fid) for fid in field_ids)
    fields_list = ", ".join(field_ids)
    return (
        "Заполни следующие поля заявки НВОС на основе контекста из документов.\n\n"
        "## Контекст из документов\n\n"
        + context_block
        + "\n\n---\n\n## Поля для заполнения\n\n"
        + field_headers
        + f"\n\n---\n\nИзвлеки значения для полей: {fields_list}.\n"
        + _PROMPT_FOOTER
    )
