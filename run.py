"""
run.py — точка входа пайплайна НВОС.

Конфигурация сосредоточена в разделе CONFIG ниже.
Для переключения между vLLM и OpenAI достаточно изменить LLM_BASE_URL и LLM_API_KEY.
"""

import json
from pathlib import Path

import instructor
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

from config.group_config import ACTIVE_GROUPING
from pipeline import run_pipeline
from retrieval.store import VectorStoreRegistry
from run_log.run_logger import RunLogger


# =============================================================================
# CONFIG — все настройки запуска в одном месте
# =============================================================================

# --- LLM (vLLM, OpenAI-совместимый API) ---
LLM_BASE_URL  = "http://localhost:8000/v1"   # адрес запущенного vLLM сервера
LLM_API_KEY   = "EMPTY"                      # vLLM не требует настоящий ключ
LLM_MODEL     = "Qwen/Qwen2.5-7B-Instruct"   # имя модели, переданное в start_vllm.sh

# --- Embed-модель (та же, что использовалась при построении ChromaDB) ---
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_DEVICE     = "cuda:0"

# --- Векторная база ---
CHROMA_DEFAULT_PATH = (
    "/mnt/localhdd/externalssd_data/Artem/ReportGenLLM/notebooks/"
    "test_db/chroma_db_TestDoc_3"
)

# --- JSONL-корпус для BM25 (каждая строка — {"page_content": ..., "metadata": ...}) ---
# Если файл недоступен — BM25 будет отключён, поиск работает только по вектору.
CORPUS_PATH: str | None = (
    "/mnt/localhdd/externalssd_data/Artem/ReportGenLLM/notebooks/test_db/postprocessed_data_3.jsonl"
)
# BM25 строится только по Text-чанкам (исключаем TableRow и AllTableMD)
BM25_RECORD_TYPES: list[str] = ["Text"]

# --- Параметры ретрива ---
# Режим: "per_field" — свои чанки для каждого поля
#        "group_deduplicated" — общий дедуплицированный пул на всю группу
RETRIEVAL_MODE = "group_deduplicated"
K_TEXT   = 10   # Text-чанков на каждый RAG-запрос
K_TABLE  = 6    # TableRow-чанков на каждый RAG-запрос
GROUP_K  = 20   # итоговый пул для group_deduplicated

# --- Логи ---
LOGS_DIR = "logs"


# =============================================================================
# Сборка компонентов
# =============================================================================

def build_embed_model() -> HuggingFaceEmbeddings:
    print(f"[Run] Загрузка embed-модели '{EMBED_MODEL_NAME}' на {EMBED_DEVICE}...")
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": EMBED_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def build_registry(embed_model) -> VectorStoreRegistry:
    registry = VectorStoreRegistry(embed_model)

    # Регистрируем текущую базу как "default".
    # В будущем добавляем дополнительные коллекции:
    #   registry.register("emissions", "/path/to/chroma_emissions", corpus_path="...")
    #   registry.register("waste",     "/path/to/chroma_waste",     corpus_path="...")
    registry.register(
        name="default",
        persist_directory=CHROMA_DEFAULT_PATH,
        corpus_path=CORPUS_PATH,
        bm25_record_types=BM25_RECORD_TYPES,
    )

    return registry


def build_llm_client() -> instructor.Instructor:
    print(f"[Run] LLM: {LLM_MODEL} @ {LLM_BASE_URL}")
    openai_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    # Mode.JSON — обязательно для vLLM (tool_calls не поддерживается по умолчанию)
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


# =============================================================================
# Точка входа
# =============================================================================

def main():
    embed_model = build_embed_model()
    registry    = build_registry(embed_model)
    llm_client  = build_llm_client()

    logger = RunLogger(
        model_name=LLM_MODEL,
        embed_model_name=EMBED_MODEL_NAME,
        grouping_strategy=ACTIVE_GROUPING,
        base_dir=LOGS_DIR,
    )

    result = run_pipeline(
        registry=registry,
        llm_client=llm_client,
        model_name=LLM_MODEL,
        logger=logger,
        retrieval_mode=RETRIEVAL_MODE,
        k_text=K_TEXT,
        k_table=K_TABLE,
        group_k=GROUP_K,
    )

    # Сохраняем итоговый результат рядом с summary
    output_path = logger.run_dir / "result.json"
    serializable = {
        fid: (value.model_dump() if value is not None else None)
        for fid, value in result.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"[Run] Результат сохранён: {output_path}")
    return result


if __name__ == "__main__":
    main()
