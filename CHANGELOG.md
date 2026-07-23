# Changelog

All notable changes to **DocuMind AI – RAG Knowledge Retrieval Platform** will be documented in this file.

## v1.0.0 (Initial Public Release)

Initial public release of the enterprise-grade RAG Knowledge Retrieval Platform.

### Features
- **FastAPI Core**: Asynchronous web backend with Pydantic v2 schemas and modular APIRouters.
- **LangChain RAG Pipeline**: PDF text loading (`PyPDFLoader`), recursive character chunking (`RecursiveCharacterTextSplitter`), retrieval chains (`RetrievalQA`), and prompt templating (`PromptTemplate`).
- **Persistent Vector Store**: ChromaDB vector store with HNSW Cosine Similarity indexing (`hnsw:space: cosine`).
- **HuggingFace Embeddings**: Local high-performance embeddings model (`sentence-transformers/all-MiniLM-L6-v2`).
- **Multi-Model Ollama Support**: Offline LLM synthesis supporting `llama3`, `mistral`, and `deepseek` models.
- **JWT Security & Auth**: Password hashing (`bcrypt`) and OAuth2 Bearer token issuance (`/login`, `/token`, `/register`).
- **Relational Persistence**: SQLite database via SQLAlchemy 2.0 ORM managing user accounts, document metadata, and chat query logs.
- **Containerization**: Production `Dockerfile` and `docker-compose.yml` specs.
- **Automated Testing**: Pytest unit test suite covering health, authentication, document validation, and query parsing.
