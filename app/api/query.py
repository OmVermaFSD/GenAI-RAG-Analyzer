from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.request_models import QueryRequest
from app.models.response_models import QueryResponse
from app.models.db_models import User
from app.services.auth_service import get_current_user
from app.services.rag_service import rag_service

router = APIRouter(tags=["RAG Core Query"])

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_knowledge_base(
    request: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Query the ingested document knowledge base:
    - Performs Vector Cosine Similarity retrieval in ChromaDB.
    - Synthesizes LLM answer via Ollama (llama3, mistral, or deepseek).
    - Returns structured JSON with query, generated answer, and source document chunks.
    - Logs query response time metrics and history in SQLite.
    """
    response_data = rag_service.query_knowledge_base(
        question=request.question,
        model_override=request.model,
        db=db,
        user=current_user
    )
    return QueryResponse(**response_data)
