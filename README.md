# DocuMind AI – RAG Knowledge Retrieval Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.1-009688.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.11-1C3C3C.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5.5-orange.svg)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Llama3%20%7C%20Mistral%20%7C%20DeepSeek-black.svg)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

### Project Status
- ✅ **Production Ready**
- ✅ **Resume Aligned**
- ✅ **Docker Supported**
- ✅ **Unit Tested**

**DocuMind AI – RAG Knowledge Retrieval Platform** is an enterprise-ready Retrieval-Augmented Generation (RAG) backend constructed with **Python**, **FastAPI**, **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, **Ollama**, **SQLite**, and **JWT Authentication**.

The system enables secure PDF document ingestion, recursive text chunking, persistent vector similarity retrieval (cosine distance), and context-aware LLM answer synthesis across multiple local LLMs.

---

## 🏗️ System Architecture

```
                                      +------------------------------------------+
                                      |            Client Application            |
                                      +--------------------+---------------------+
                                                           |
                                                JWT Bearer Token Auth
                                                           |
                                           +---------------+---------------+
                                           |                               |
                                POST /upload (PDF)                   POST /query (JSON)
                                           v                               v
                                +--------------------------------------------------+
                                |               FastAPI Web Server                 |
                                |            (Modular Router Architecture)         |
                                +---------+------------------------------+---------+
                                          |                              |
                   PDF Parsing & Chunking |                              | Vector Search & LLM Prompting
                                          v                              v
+---------------------------------------------------+  +---------------------------------------------------+
| Document Service (PyPDFLoader)                    |  | RAG Service (RetrievalQA Chain)                   |
| - RecursiveCharacterTextSplitter                  |  | - PromptTemplate Engine                           |
|   (chunk_size=1000, chunk_overlap=200)            |  | - Response Duration Logger                        |
+-------------------------+-------------------------+  +-------------------------+-------------------------+
                          |                                                    |
                          v                                                    v
+---------------------------------------------------+  +---------------------------------------------------+
| HuggingFace Embedding Service                     |  | Ollama LLM Service                                |
| (sentence-transformers/all-MiniLM-L6-v2)          |  | (Configurable: llama3 | mistral | deepseek)       |
+-------------------------+-------------------------+  +-------------------------+-------------------------+
                          |                                                    |
                          +-------------------------+--------------------------+
                                                    |
                                                    v
+---------------------------------------------------+  +---------------------------------------------------+
| Persistent ChromaDB Store                         |  | SQLite Relational DB                              |
| (data/chroma_db - Cosine Similarity Index)        |  | (Users | Chat History | Upload Metadata)          |
+---------------------------------------------------+  +---------------------------------------------------+
```

---

## 📁 Repository Structure

```
.
├── app/
│   ├── main.py                  # FastAPI Application & Lifespan Initialization
│   ├── api/                     # REST API APIRouters
│   │   ├── auth.py              # User Registration, /login, & /token JWT Endpoints
│   │   ├── upload.py            # PDF File Upload & Processing Endpoint
│   │   ├── query.py             # RAG Vector Search & LLM Synthesis Endpoint
│   │   └── health.py            # System & DB Health Check Endpoints
│   ├── services/                # Core Business Logic Layer
│   │   ├── rag_service.py       # LangChain RetrievalQA Chain & Execution Logic
│   │   ├── document_service.py  # PDF Loader, Recursive Chunking & Storage
│   │   ├── embedding_service.py # HuggingFace Embeddings Model Wrapper
│   │   └── auth_service.py      # Password Hashing & JWT Validation
│   ├── vectorstore/
│   │   └── chroma_manager.py    # Persistent ChromaDB Manager (Cosine Similarity)
│   ├── llm/
│   │   └── ollama_client.py     # Ollama API Client (llama3, mistral, deepseek)
│   ├── db/
│   │   └── database.py          # SQLite Connection & Session Manager
│   ├── models/
│   │   ├── request_models.py    # Pydantic Request Schemas
│   │   ├── response_models.py   # Pydantic Response Schemas
│   │   └── db_models.py         # SQLAlchemy Database Models
│   ├── config/
│   │   └── settings.py          # Central Pydantic BaseSettings Configuration
│   └── utils/
│       ├── logger.py            # Structured Logging & Duration Trackers
│       └── exceptions.py        # Domain Exceptions & FastAPI Handlers
├── data/
│   ├── uploads/                 # Persisted Raw PDF Files
│   └── chroma_db/               # Persisted Vector DB Storage
├── tests/                       # Pytest Automated Test Suite
│   ├── test_auth.py
│   ├── test_upload.py
│   └── test_query.py
├── app.py                       # Backward-Compatibility Entrypoint
├── Dockerfile                   # Production Container Image Spec
├── docker-compose.yml           # Docker Orchestration Configuration
├── requirements.txt             # Pinned Python Dependencies
└── .env.example                 # Environment Variable Blueprint
```

---

## 🌟 Key Technical Features

1. **Modular Architecture & Dependency Injection**: Clean separation of API controllers, service handlers, database models, and vector storage engines adhering to SOLID principles.
2. **Recursive Text Chunking**: Leverages `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200` to preserve logical sentence boundaries.
3. **Local Vector Embeddings**: Utilizes `sentence-transformers/all-MiniLM-L6-v2` for 384-dimensional dense vector embeddings.
4. **Persistent Vector Store**: ChromaDB instance running Cosine Similarity indexing persisted under `./data/chroma_db`.
5. **Multi-Model Ollama Support**: Fully configurable support for `llama3`, `mistral`, and `deepseek` local LLMs.
6. **JWT Authentication & Security**: Password hashing via `bcrypt` and JWT bearer token authorization protecting ingestion and query routes.
7. **Relational Metadata Storage**: SQLite database storing user accounts, uploaded document metadata, and complete chat history with response latency tracking.
8. **Automated Testing**: Comprehensive unit & integration tests covering authorization, document validation, and query parsing.

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites

- **Python**: `3.10+`
- **Ollama**: Download and start Ollama locally ([Ollama Website](https://ollama.com/download))

Pull your model of choice:
```bash
ollama pull llama3
# or
ollama pull mistral
# or
ollama pull deepseek-r1
```

### 2. Environment Configuration

Clone the repository and prepare `.env`:
```bash
cp .env.example .env
```

### 3. Installation & Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Running Locally

Start the server using Uvicorn:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- **Interactive API Documentation**: `http://localhost:8000/docs`
- **ReDoc Specification**: `http://localhost:8000/redoc`

---

## 🐳 Docker Deployment

To launch the full backend alongside Ollama using Docker Compose:

```bash
docker-compose up --build -d
```

Check running services:
```bash
docker-compose ps
```

---

## 📡 API Usage Guide

### 1. Register User & Obtain JWT Token

**Register**:
```bash
curl -X POST "http://localhost:8000/register" \
     -H "Content-Type: application/json" \
     -d '{"username": "senior_dev", "password": "securepassword123"}'
```

**Login & Get JWT Bearer Token**:
```bash
curl -X POST "http://localhost:8000/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "senior_dev", "password": "securepassword123"}'
```
*Response*:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

---

### 2. Upload PDF Document

```bash
curl -X POST "http://localhost:8000/upload" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -F "file=@/path/to/your_document.pdf"
```

*Response*:
```json
{
  "filename": "your_document.pdf",
  "status": "success",
  "total_chunks": 24,
  "message": "Successfully ingested and indexed 24 vector chunks into ChromaDB."
}
```

---

### 3. Query Document Knowledge Base

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
           "question": "What are the core capabilities of the architecture?",
           "model": "llama3"
         }'
```

*Response*:
```json
{
  "query": "What are the core capabilities of the architecture?",
  "answer": "The core capabilities include recursive text chunking, persistent vector storage via ChromaDB, and local LLM response synthesis.",
  "source_documents": [
    {
      "page_content": "Key system capabilities involve multi-page PDF processing using PyPDFLoader...",
      "metadata": {
        "page": 0,
        "source_filename": "your_document.pdf"
      }
    }
  ],
  "response_time_ms": 1420.5
}
```

---

### 4. Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

*Response*:
```json
{
  "status": "healthy",
  "vector_store_initialized": true,
  "total_documents": 24,
  "ollama_model": "llama3",
  "database_status": "connected"
}
```

---

## ⚙️ Environment Variables

The application can be configured via environment variables defined in a `.env` file:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `"DocuMind AI – RAG Knowledge Retrieval Platform"` | Display title of the FastAPI application |
| `DEBUG` | `True` | Enables development debug logging and reload mode |
| `HOST` | `"0.0.0.0"` | Binding network host address |
| `PORT` | `8000` | HTTP web server port |
| `OLLAMA_MODEL` | `"llama3"` | Default Ollama model (`llama3`, `mistral`, `deepseek`) |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama service endpoint base URL |
| `EMBEDDING_MODEL_NAME` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace embedding model ID |
| `CHROMA_PERSIST_DIR` | `"./data/chroma_db"` | Persistent vector database directory path |
| `UPLOAD_DIR` | `"./data/uploads"` | Persistent PDF document storage directory |
| `CHUNK_SIZE` | `1000` | Text chunk character length |
| `CHUNK_OVERLAP` | `200` | Overlap character length between chunks |
| `SQLITE_DB_URL` | `"sqlite:///./data/app.db"` | Relational database connection string |
| `JWT_SECRET_KEY` | `"super-secret-key-change-in-production-123456789"` | Cryptographic secret key for signing JWT tokens |
| `JWT_ALGORITHM` | `"HS256"` | JWT signing algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Bearer token validity duration in minutes |

---

## 🛠️ Technology Stack

| Layer | Component | Purpose |
| :--- | :--- | :--- |
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance ASGI web framework with Pydantic v2 validation |
| **Orchestration** | [LangChain](https://www.langchain.com/) | RetrievalQA chain, PromptTemplates, and splitters |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | Persistent vector database with Cosine Similarity indexing |
| **Embeddings** | [HuggingFace](https://huggingface.co/) | `sentence-transformers/all-MiniLM-L6-v2` dense vector model |
| **Local LLM** | [Ollama](https://ollama.com/) | Privacy-first offline context synthesis engine (`llama3`, `mistral`, `deepseek`) |
| **Relational DB** | [SQLite](https://www.sqlite.org/) + [SQLAlchemy](https://www.sqlalchemy.org/) | User management, document metadata, and query history storage |
| **Security** | [PyJWT](https://pyjwt.readthedocs.io/) + [Passlib](https://passlib.readthedocs.io/) | OAuth2 Bearer authentication and bcrypt password hashing |
| **Containerization** | [Docker](https://www.docker.com/) | Docker & Docker Compose deployment specs |

---

## 📸 API Documentation & Screenshots

### Interactive Swagger UI (`/docs`)
Access the built-in OpenAPI interactive documentation interface at `http://localhost:8000/docs`:

```
+-------------------------------------------------------------------------------+
|  DocuMind AI – RAG Knowledge Retrieval Platform  [v2.0.0]                     |
|  [ OpenAPI Specification: /openapi.json ]                                     |
+-------------------------------------------------------------------------------+
|  POST  /register    Register a new user in SQLite database                   |
|  POST  /login       Authenticate user and return JWT Access Token             |
|  POST  /token       OAuth2 compatible token login for Swagger Authorization   |
|  POST  /upload      Upload and ingest PDF document into ChromaDB              |
|  POST  /query       Execute RAG vector similarity search and LLM synthesis    |
|  GET   /health      Returns system, vector store, and SQLite status           |
+-------------------------------------------------------------------------------+
```

---

## 🧪 Running Tests

Execute the pytest suite:

```bash
pytest tests/ -v
```

---

## 🔮 Future Enhancements

- **Hybrid Search**: Integrate BM25 Keyword Search alongside Cosine Vector Similarity (Hybrid Retrieval).
- **Asynchronous Ingestion**: Offload PDF processing tasks to Celery / Redis background task queues for large files.
- **RAG Evaluation**: Implement Ragas / TruLens evaluation benchmarks to score answer faithfulness and relevance.

---

## 📌 Versioning
Current Release: **`v1.0.0`**

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and branch naming standards.

---

## 📜 License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.


