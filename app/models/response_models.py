from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class IngestResponse(BaseModel):
    filename: str = Field(..., description="Name of the ingested PDF file")
    status: str = Field(..., description="Ingestion operation status")
    total_chunks: int = Field(..., description="Total text chunks generated and stored")
    message: str = Field(..., description="Detailed status message")

class SourceChunk(BaseModel):
    page_content: str = Field(..., description="Extracted text chunk snippet")
    metadata: Dict[str, Any] = Field(..., description="Document metadata (page, source file, etc.)")

class QueryResponse(BaseModel):
    query: str = Field(..., description="Original question submitted")
    answer: str = Field(..., description="Synthesized LLM answer based on retrieved context")
    source_documents: List[SourceChunk] = Field(..., description="Source context chunks used for answer generation")
    response_time_ms: Optional[float] = Field(None, description="Model query processing time in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    vector_store_initialized: bool = Field(..., description="Whether ChromaDB contains indexed documents")
    total_documents: int = Field(..., description="Total vector count in ChromaDB")
    ollama_model: str = Field(..., description="Currently active Ollama LLM model")
    database_status: str = Field(..., description="SQLite database connectivity status")

class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT Bearer Token")
    token_type: str = Field("bearer", description="Token type")

class UserResponse(BaseModel):
    id: int = Field(..., description="User ID")
    username: str = Field(..., description="Username")
