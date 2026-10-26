"""LangChain Embeddings wrapper around fastembed (local, free, no API key)."""

from collections.abc import Sequence

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class FastEmbedEmbeddings(Embeddings):
    """Runs a small multilingual embedding model locally via ONNX.

    The model (~220 MB) is downloaded on first use and cached by fastembed.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: TextEmbedding | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:  # lazy: only load the model when it is actually needed
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.embed(list(texts), batch_size=self.batch_size)
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
