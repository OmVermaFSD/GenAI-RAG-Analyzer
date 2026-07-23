from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from app.config.settings import settings
from app.utils.logger import logger

class EmbeddingService:
    """Service wrapper for HuggingFace Embedding models."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Returns initialized HuggingFaceEmbeddings instance."""
        if self._embeddings is None:
            logger.info(f"Loading HuggingFace embeddings model: '{self.model_name}'")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings

# Singleton instance
embedding_service = EmbeddingService()
