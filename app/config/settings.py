from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "DocuMind AI – RAG Knowledge Retrieval Platform"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    OLLAMA_MODEL: str = "llama3"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploads"

    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    SQLITE_DB_URL: str = "sqlite:///./data/app.db"

    JWT_SECRET_KEY: str = "super-secret-key-change-in-production-123456789"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

