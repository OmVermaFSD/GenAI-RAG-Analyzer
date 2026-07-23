import os
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from app.config.settings import settings
from app.utils.logger import logger
from app.utils.exceptions import VectorStoreError

class ChromaManager:
    """Manages persistent ChromaDB vector store initialization, retrieval, and document addition."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.vector_store: Optional[Chroma] = None

    def initialize(self, embedding_function) -> Chroma:
        """Initialize or connect to persistent ChromaDB storage using specified embedding function."""
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            logger.info(f"Connecting to persistent ChromaDB at '{self.persist_dir}' with Cosine Similarity...")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embedding_function,
                collection_metadata={"hnsw:space": "cosine"}
            )
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}", exc_info=True)
            raise VectorStoreError(f"ChromaDB initialization failed: {str(e)}")

    def add_documents(self, documents: List[Document]) -> int:
        """Persist document chunks into ChromaDB."""
        if not self.vector_store:
            raise VectorStoreError("ChromaDB is not initialized.")
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Successfully indexed {len(documents)} document chunks into ChromaDB.")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}", exc_info=True)
            raise VectorStoreError(f"Document indexing failed: {str(e)}")

    def get_retriever(self, k: int = 4):
        """Returns similarity retriever configured for top-k cosine similarity."""
        if not self.vector_store:
            raise VectorStoreError("ChromaDB is not initialized.")
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def get_document_count(self) -> int:
        """Returns total chunk/document count stored in ChromaDB collection."""
        if self.vector_store and hasattr(self.vector_store, "_collection"):
            try:
                return self.vector_store._collection.count()
            except Exception:
                return 0
        return 0

# Global singleton instance
chroma_manager = ChromaManager()
