"""
DocuMind AI - Production-Grade RAG Knowledge Retrieval Backend
FastAPI + LangChain + ChromaDB + HuggingFace Embeddings + Ollama (Llama 3) + PyPDF
"""

import os
import shutil
import tempfile
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain & Vector DB Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DocuMind-AI")

# Directory Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_store")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ============================================================================
# Pydantic Request & Response Schemas
# ============================================================================

class IngestResponse(BaseModel):
    filename: str = Field(..., description="Name of the ingested PDF file")
    status: str = Field(..., description="Ingestion operation status")
    total_chunks: int = Field(..., description="Total text chunks generated and stored")
    message: str = Field(..., description="Detailed status message")

class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=3, 
        description="Natural language question to query against ingested documents",
        examples=["What is the main objective of this document?"]
    )

class SourceChunk(BaseModel):
    page_content: str = Field(..., description="Extracted text chunk snippet")
    metadata: Dict[str, Any] = Field(..., description="Document metadata (page, source file, etc.)")

class QueryResponse(BaseModel):
    query: str = Field(..., description="Original question submitted")
    answer: str = Field(..., description="Synthesized LLM answer based on retrieved context")
    source_documents: List[SourceChunk] = Field(..., description="Source context chunks used for answer generation")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    vector_store_initialized: bool = Field(..., description="Whether ChromaDB contains indexed documents")
    total_documents: int = Field(..., description="Total vector count in ChromaDB")


# ============================================================================
# RAG Engine Service Manager
# ============================================================================

class RAGEngine:
    def __init__(self):
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[Chroma] = None
        self.llm: Optional[Ollama] = None

    def initialize(self):
        """Initialize HuggingFace Embeddings and loading/creating Chroma vector database."""
        logger.info(f"Initializing HuggingFace embeddings model: {EMBEDDING_MODEL_NAME}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        logger.info(f"Connecting to ChromaDB store at: {CHROMA_PERSIST_DIR}")
        self.vector_store = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Initializing Ollama LLM with model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")
        self.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1
        )

    def ingest_pdf(self, file_path: str, filename: str) -> int:
        """Process PDF: extract text, chunk text recursively, generate embeddings & persist to ChromaDB."""
        if not self.vector_store or not self.embeddings:
            raise RuntimeError("RAG Engine is not properly initialized.")

        logger.info(f"Loading document: {filename} from temp path: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise ValueError(f"No readable text found in PDF: {filename}")

        # Update metadata to include original filename
        for doc in documents:
            doc.metadata["source_filename"] = filename

        # Split document recursively
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} text chunks from document '{filename}'")

        # Store chunks in ChromaDB
        self.vector_store.add_documents(chunks)
        logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB.")

        return len(chunks)

    def query(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Perform vector search cosine similarity retrieval and Ollama LLM synthesis."""
        if not self.vector_store or not self.llm:
            raise RuntimeError("RAG Engine is not initialized.")

        # Check if vector store has any documents
        collection = self.vector_store._collection
        if collection.count() == 0:
            raise ValueError("Vector database is empty. Please ingest at least one PDF document before querying.")

        # Set up retriever using Cosine Similarity
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

        custom_prompt_template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer concise and well-explained.

Context:
{context}

Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        logger.info(f"Executing RAG query: '{question}'")
        result = qa_chain.invoke({"query": question})

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        formatted_sources = [
            SourceChunk(
                page_content=doc.page_content,
                metadata=doc.metadata
            )
            for doc in source_docs
        ]

        return {
            "query": question,
            "answer": answer,
            "source_documents": formatted_sources
        }

    def get_document_count(self) -> int:
        if self.vector_store and hasattr(self.vector_store, "_collection"):
            return self.vector_store._collection.count()
        return 0


# Global Engine Instance
rag_engine = RAGEngine()


# ============================================================================
# FastAPI Lifespan & Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and Shutdown Lifespan Manager."""
    logger.info("Starting up DocuMind AI RAG Backend Service...")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    try:
        rag_engine.initialize()
    except Exception as e:
        logger.error(f"Error during RAG Engine startup initialization: {str(e)}")
    yield
    logger.info("Shutting down DocuMind AI service.")

app = FastAPI(
    title="DocuMind AI - Production RAG Backend",
    description="High-performance RAG Knowledge Retrieval System powered by FastAPI, LangChain, ChromaDB, HuggingFace, and Ollama (Llama 3).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/", tags=["Health Check"])
async def root():
    return {
        "app": "DocuMind AI - RAG Knowledge Retrieval Backend",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    """Returns application health and ChromaDB document store readiness."""
    total_docs = rag_engine.get_document_count()
    return HealthResponse(
        status="healthy",
        vector_store_initialized=total_docs > 0,
        total_documents=total_docs
    )

@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["RAG Core"])
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Ingest a PDF document:
    - Extracts raw text using PyPDF.
    - Chunks text recursively (1000 size, 200 overlap).
    - Generates vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
    - Persists embeddings into ChromaDB (`./chroma_db_store`).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

    try:
        # Save uploaded file temporarily for PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        try:
            total_chunks = rag_engine.ingest_pdf(temp_file_path, file.filename)
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return IngestResponse(
            filename=file.filename,
            status="success",
            total_chunks=total_chunks,
            message=f"Successfully ingested and indexed {total_chunks} vector chunks into ChromaDB."
        )

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Ingestion failed for file '{file.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during PDF processing: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, tags=["RAG Core"])
async def query_knowledge_base(request: QueryRequest):
    """
    Query the ingested document store:
    - Performs Vector Cosine Similarity retrieval.
    - Synthesizes LLM answer via Ollama (Llama 3).
    - Returns structured JSON with query, generated answer, and source document chunks.
    """
    try:
        response_data = rag_engine.query(request.question)
        return QueryResponse(**response_data)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re))
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
