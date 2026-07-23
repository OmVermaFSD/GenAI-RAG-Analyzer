from fastapi import APIRouter, UploadFile, File, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.response_models import IngestResponse
from app.models.db_models import User
from app.services.auth_service import get_current_user
from app.services.document_service import document_service

router = APIRouter(tags=["Document Ingestion"])

@router.post("/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def upload_pdf_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and Ingest a PDF document:
    - Extracts raw text using PyPDF.
    - Chunks text recursively (chunk_size=1000, chunk_overlap=200).
    - Generates vector embeddings via HuggingFace `all-MiniLM-L6-v2`.
    - Persists vector chunks into persistent ChromaDB.
    - Records upload metadata into SQLite.
    """
    total_chunks = document_service.process_and_ingest_pdf(file, db=db, user=current_user)
    return IngestResponse(
        filename=file.filename,
        status="success",
        total_chunks=total_chunks,
        message=f"Successfully ingested and indexed {total_chunks} vector chunks into ChromaDB."
    )
