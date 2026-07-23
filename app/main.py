import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.db.database import init_db
from app.services.embedding_service import embedding_service
from app.vectorstore.chroma_manager import chroma_manager
from app.utils.logger import logger
from app.utils.exceptions import RAGException, rag_exception_handler
from app.api import health, auth, upload, query

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application Startup and Shutdown Lifespan Manager."""
    logger.info("==================================================")
    logger.info(f"Starting {settings.APP_NAME}...")
    logger.info("==================================================")
    
    # Create required data directories
    os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # Initialize SQLite tables
    try:
        init_db()
    except Exception as e:
        logger.error(f"Error during SQLite DB initialization: {str(e)}")

    # Initialize Vector Store and Embeddings
    try:
        embeddings = embedding_service.get_embeddings()
        chroma_manager.initialize(embeddings)
    except Exception as e:
        logger.error(f"Error during vector store initialization: {str(e)}")

    yield
    logger.info(f"Shutting down {settings.APP_NAME}.")

app = FastAPI(
    title=settings.APP_NAME,
    description="Production-Ready Retrieval-Augmented Generation (RAG) Backend built with FastAPI, LangChain, ChromaDB, HuggingFace, Ollama, SQLite, and JWT Auth.",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handlers
app.add_exception_handler(RAGException, rag_exception_handler)

# Include Routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(upload.router)
app.include_router(query.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
