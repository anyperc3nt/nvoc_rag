"""
extract_group — вызов LLM через instructor для группы полей.

Ключевая идея: для каждой группы динамически строится Pydantic-модель GroupResponse
через pydantic.create_model. Каждое поле — Optional[FieldModel | NotFound].
Instructor возвращает частично заполненную группу (некоторые поля могут быть NotFound).
"""

from __future__ import annotations

import json
from typing import Annotated, Optional, Union, get_args, get_origin

import instructor
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from config.field_config import FIELD_CONFIG
from extraction.schemas import EXTRACTION_TYPE_MAP, NotFound


def build_group_schema(field_ids: list[str]) -> type[BaseModel]:
    """
    Динамически строит Pydantic-модель для группы полей.

    Каждое поле группы становится Optional[FieldModel | NotFound].
    Имя поля — ID поля из FIELD_CONFIG (например, "ПОЛЕ_1").

    Args:
        field_ids: Список ID полей в группе.

    Returns:
        Динамически созданный класс GroupResponse(BaseModel).
    """
    field_defs: dict[str, tuple] = {}

    for fid in field_ids:
        cfg = FIELD_CONFIG[fid]
        extraction_type = cfg["extraction_type"]
        inner_model = EXTRACTION_TYPE_MAP[extraction_type]

        # Optional[Union[inner_model, NotFound]] с default=None
        field_defs[fid] = (
            Optional[Union[inner_model, NotFound]],
            FieldInfo(default=None, description=cfg["description"]),
        )

    return create_model("GroupResponse", **field_defs)


def extract_group(
    llm_client: instructor.Instructor,
    model_name: str,
    field_ids: list[str],
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 2,
) -> tuple[str, dict[str, BaseModel | None]]:
    """
    Вызывает LLM для группы полей и возвращает распаршенный результат.

    Args:
        llm_client:    Instructor-клиент (обёртка над OpenAI-совместимым API).
        model_name:    Имя модели (например, "Qwen/Qwen2.5-72B-Instruct").
        field_ids:     Список ID полей в группе.
        system_prompt: Системный промпт.
        user_prompt:   Пользовательский промпт (контекст + описания полей).
        max_retries:   Количество попыток при ошибке валидации.

    Returns:
        (llm_response_raw, parsed_dict)
        llm_response_raw — сырой JSON-ответ в виде строки
        parsed_dict      — dict[field_id -> FieldModel | NotFound | None]
    """
    group_schema = build_group_schema(field_ids)

    response: BaseModel = llm_client.chat.completions.create(
        model=model_name,
        response_model=group_schema,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )

    # Сериализуем сырой ответ для логов
    llm_response_raw = response.model_dump_json(indent=2)

    # Разбиваем по полям
    parsed_dict: dict[str, BaseModel | None] = {}
    response_data = response.model_dump()
    for fid in field_ids:
        raw_value = response_data.get(fid)
        if raw_value is None:
            parsed_dict[fid] = None
            continue

        # Определяем тип по ключам словаря: NotFound имеет только "reason"
        if set(raw_value.keys()) == {"reason"}:
            parsed_dict[fid] = NotFound(**raw_value)
        else:
            extraction_type = FIELD_CONFIG[fid]["extraction_type"]
            model_cls = EXTRACTION_TYPE_MAP[extraction_type]
            parsed_dict[fid] = model_cls(**raw_value)

    return llm_response_raw, parsed_dict


def parsed_to_loggable(parsed_dict: dict[str, BaseModel | None]) -> dict:
    """
    Конвертирует dict[field_id -> Pydantic-объект] в plain dict для логов.
    None и NotFound сохраняются явно.
    """
    result = {}
    for fid, value in parsed_dict.items():
        if value is None:
            result[fid] = None
        else:
            result[fid] = value.model_dump()
    return result
