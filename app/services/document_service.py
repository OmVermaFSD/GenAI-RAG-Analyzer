import os
import shutil
from typing import Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import settings
from app.utils.logger import logger
from app.utils.exceptions import DocumentIngestionError
from app.vectorstore.chroma_manager import chroma_manager
from app.models.db_models import DocumentMetadata, User

class DocumentService:
    """Service handling PDF document saving, text extraction, recursive chunking, ChromaDB indexing, and SQLite metadata recording."""

    def __init__(self, upload_dir: Optional[str] = None):
        self.upload_dir = upload_dir or settings.UPLOAD_DIR
        os.makedirs(self.upload_dir, exist_ok=True)

    def process_and_ingest_pdf(self, file: UploadFile, db: Optional[Session] = None, user: Optional[User] = None) -> int:
        """Saves PDF file, splits into text chunks recursively, adds to vector store, and records metadata."""
        filename = file.filename
        if not filename.lower().endswith(".pdf"):
            raise DocumentIngestionError("Invalid file format. Only PDF files (.pdf) are supported.")

        saved_file_path = os.path.join(self.upload_dir, filename)

        try:
            logger.info(f"Saving uploaded file '{filename}' to '{saved_file_path}'")
            with open(saved_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save file '{filename}': {str(e)}", exc_info=True)
            raise DocumentIngestionError(f"Failed to save uploaded PDF file: {str(e)}")

        try:
            logger.info(f"Extracting text from PDF: '{filename}'")
            loader = PyPDFLoader(saved_file_path)
            documents = loader.load()

            if not documents:
                raise DocumentIngestionError(f"No readable text found in PDF document '{filename}'")

            for doc in documents:
                doc.metadata["source_filename"] = filename

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Generated {len(chunks)} text chunks for '{filename}'")

            # Add to vector store
            total_chunks = chroma_manager.add_documents(chunks)

            # Record in SQLite if session provided
            if db:
                doc_record = DocumentMetadata(
                    filename=filename,
                    file_path=saved_file_path,
                    total_chunks=total_chunks,
                    uploaded_by=user.id if user else None
                )
                db.add(doc_record)
                db.commit()
                db.refresh(doc_record)

            return total_chunks

        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during PDF ingestion: {str(e)}", exc_info=True)
            raise DocumentIngestionError(f"Document ingestion failed: {str(e)}")

# Singleton instance
document_service = DocumentService()
