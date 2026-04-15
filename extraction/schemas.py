"""
Pydantic-схемы для структурированного вывода LLM (через instructor).

Каждая схема соответствует одному extraction_type из field_config.py:
    SHORT_STRING  → ShortStringField
    LONG_STRING   → LongStringField
    DATE          → DateField
    TABLE         → TableField
    BOOLEAN       → BooleanField
    REQUISITES    → RequisitesField
    COORDINATES   → CoordinatesField

Во всех схемах присутствуют:
    confidence      — уверенность модели [0, 1]
    source_fragment — цитата из документа, откуда взято значение

NotFound возвращается, когда данные не найдены или неоднозначны.
"""

from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# NotFound — универсальный ответ при отсутствии данных
# =============================================================================

class NotFound(BaseModel):
    reason: str = Field(
        description=(
            "Объяснение, почему данные не найдены: отсутствуют в контексте, "
            "неоднозначны, или поле не применимо к данному объекту."
        )
    )


# =============================================================================
# Простые скалярные типы
# =============================================================================

class ShortStringField(BaseModel):
    """Одна строка или число (номер, название, код и т.п.)."""
    value: str = Field(description="Извлечённое значение.")
    confidence: float = Field(ge=0.0, le=1.0, description="Уверенность [0–1].")
    source_fragment: str = Field(
        description="Дословная цитата из документа, подтверждающая значение."
    )


class LongStringField(BaseModel):
    """Абзац описательного текста."""
    value: str = Field(description="Извлечённый текст (может быть несколько предложений).")
    confidence: float = Field(ge=0.0, le=1.0)
    source_fragment: str = Field(
        description="Дословная цитата или краткое указание на источник."
    )


class DateField(BaseModel):
    """Дата в формате DD.MM.YYYY или YYYY (только год)."""
    value: str = Field(
        description="Дата в формате DD.MM.YYYY или YYYY (только год)."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    source_fragment: str


class BooleanField(BaseModel):
    """Логическое значение: да / нет."""
    value: bool = Field(description="True = да, False = нет.")
    confidence: float = Field(ge=0.0, le=1.0)
    source_fragment: str


# =============================================================================
# Реквизиты документа
# =============================================================================

class RequisitesField(BaseModel):
    """
    Реквизиты разрешительного или нормативного документа:
    номер, дата выдачи, выдавший орган.
    """
    document_number: Optional[str] = Field(
        default=None,
        description="Номер документа (регистрационный, реестровый и т.п.)."
    )
    issue_date: Optional[str] = Field(
        default=None,
        description="Дата выдачи/регистрации в формате DD.MM.YYYY."
    )
    issuing_authority: Optional[str] = Field(
        default=None,
        description="Наименование органа, выдавшего документ."
    )
    expiry_date: Optional[str] = Field(
        default=None,
        description="Срок действия (дата окончания) в формате DD.MM.YYYY, если указан."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    source_fragment: str


# =============================================================================
# Координаты
# =============================================================================

class CoordinatePoint(BaseModel):
    """Одна точка в геодезических или прямоугольных координатах."""
    source_id: Optional[str] = Field(
        default=None,
        description="Идентификатор источника/объекта, к которому относятся координаты."
    )
    latitude: Optional[str] = Field(
        default=None,
        description="Широта (с.ш.) или координата X — строкой для сохранения точности."
    )
    longitude: Optional[str] = Field(
        default=None,
        description="Долгота (в.д.) или координата Y — строкой."
    )
    coordinate_system: Optional[str] = Field(
        default=None,
        description="Система координат (например, WGS-84, МСК-61)."
    )


class CoordinatesField(BaseModel):
    """Список координатных точек (один или несколько объектов)."""
    points: list[CoordinatePoint] = Field(
        description="Одна или несколько точек."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    source_fragment: str


# =============================================================================
# Таблица (универсальная — подходит для всех TABLE-полей)
# =============================================================================

class TableField(BaseModel):
    """
    Универсальная таблица.

    Каждая строка — словарь column_name → value.
    Имена колонок LLM выбирает самостоятельно на основании описания поля
    и пояснений из `notes` (поданных в промпт).

    Примеры строк для разных полей:
      ПОЛЕ_32 (масса выбросов):
        {"номер_источника": "0001", "код_зв": "2908", "наименование_зв": "Азота диоксид", "масса_т_год": "0.45"}
      ПОЛЕ_47 (виды отходов):
        {"наименование_отхода": "...", "код_фкко": "...", "класс_опасности": "III", "количество_т_год": "12.5"}
    """
    rows: list[dict[str, str]] = Field(
        description=(
            "Список строк таблицы. Каждая строка — словарь: "
            "ключи — имена колонок (по описанию поля), значения — строки."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0)
    source_fragment: str = Field(
        description="Цитата или ссылка на раздел документа, откуда взята таблица."
    )


# =============================================================================
# Маппинг extraction_type → Pydantic-класс (используется в extractor.py)
# =============================================================================

EXTRACTION_TYPE_MAP: dict[str, type[BaseModel]] = {
    "short_string": ShortStringField,
    "long_string":  LongStringField,
    "date":         DateField,
    "table":        TableField,
    "boolean":      BooleanField,
    "requisites":   RequisitesField,
    "coordinates":  CoordinatesField,
}
