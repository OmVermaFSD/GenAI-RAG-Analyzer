from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.database import get_db
from app.models.response_models import HealthResponse
from app.vectorstore.chroma_manager import chroma_manager
from app.llm.ollama_client import ollama_client
from app.config.settings import settings

router = APIRouter(tags=["Health & Status"])

@router.get("/", status_code=status.HTTP_200_OK)
async def root():
    return {
        "app": settings.APP_NAME,
        "status": "online",
        "docs_url": "/docs"
    }

@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(db: Session = Depends(get_db)):
    """Returns application health, ChromaDB document index count, Ollama model, and SQLite connectivity."""
    total_docs = chroma_manager.get_document_count()
    
    # Verify SQLite DB connection
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy",
        vector_store_initialized=total_docs > 0,
        total_documents=total_docs,
        ollama_model=settings.OLLAMA_MODEL,
        database_status=db_status
    )
