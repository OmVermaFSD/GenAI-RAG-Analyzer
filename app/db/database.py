import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config.settings import settings
from app.utils.logger import logger

# Ensure directory for SQLite DB exists
db_dir = os.path.dirname(settings.SQLITE_DB_URL.replace("sqlite:///", ""))
if db_dir and not os.path.exists(db_dir):
    os.makedirs(db_dir, exist_ok=True)

engine = create_engine(
    settings.SQLITE_DB_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.SQLITE_DB_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """Dependency generator for database sessions in FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    import app.models.db_models  # noqa: F401 - Register models with Base.metadata
    logger.info("Initializing SQLite database tables...")
    Base.metadata.create_all(bind=engine)

# Auto-initialize database tables on module load
init_db()


