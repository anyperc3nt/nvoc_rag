from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class VectorStoreRegistry:
    """
    Реестр векторных баз.

    Сейчас — одна БД ("default").
    В будущем — словарь коллекций по доменам:
        registry.register("emissions", "/path/to/chroma_emissions", corpus_path="...")
        registry.register("waste",     "/path/to/chroma_waste",     corpus_path="...")
    В FIELD_CONFIG поле "collection" ссылается на имена здесь.
    """

    def __init__(self, embed_model):
        self.embed_model = embed_model
        self._stores: dict[str, Chroma] = {}

        # BM25-индексы хранятся в памяти — строятся один раз при register()
        self._bm25_indexes: dict[str, BM25Okapi] = {}
        # Тексты корпуса для BM25 (нужны при поиске, чтобы вернуть сами строки)
        self._bm25_corpus: dict[str, list[str]] = {}

    def register(
        self,
        name: str,
        persist_directory: str,
        corpus_path: Optional[str] = None,
    ) -> None:
        """
        Регистрирует коллекцию.

        Args:
            name:               Имя коллекции (ключ в registry, совпадает с
                                полем "collection" в FIELD_CONFIG).
            persist_directory:  Путь к папке ChromaDB.
            corpus_path:        Путь к текстовому файлу корпуса (одна строка = один чанк).
                                Если передан — строится BM25-индекс.
        """
        self._stores[name] = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embed_model,
        )

        if corpus_path:
            texts = [
                line for line in
                Path(corpus_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self._bm25_corpus[name] = texts
            self._bm25_indexes[name] = BM25Okapi([t.split() for t in texts])
            print(f"[Registry] '{name}': BM25 построен по {len(texts)} строкам корпуса.")

        print(f"[Registry] '{name}' зарегистрирована: {persist_directory}")

    def get_store(self, name: str) -> Chroma:
        if name not in self._stores:
            raise KeyError(
                f"Коллекция '{name}' не зарегистрирована. "
                f"Доступные: {list(self._stores.keys())}"
            )
        return self._stores[name]

    def get_bm25(self, name: str) -> Optional[BM25Okapi]:
        """Вернуть BM25-индекс или None, если corpus_path не был передан."""
        return self._bm25_indexes.get(name)

    def get_bm25_corpus(self, name: str) -> list[str]:
        """Вернуть тексты корпуса для BM25-поиска."""
        return self._bm25_corpus.get(name, [])

    def vector_query(
        self,
        name: str,
        query: str,
        k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[tuple[Document, float]]:
        """Векторный поиск с оценкой релевантности (score — косинусное расстояние)."""
        store = self.get_store(name)
        return store.similarity_search_with_score(query, k=k, filter=filters)
