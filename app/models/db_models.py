from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=utc_now)

    documents = relationship("DocumentMetadata", back_populates="uploader")
    chat_histories = relationship("ChatHistory", back_populates="user")


class DocumentMetadata(Base):
    __tablename__ = "uploaded_documents_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    total_chunks = Column(Integer, nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    upload_timestamp = Column(DateTime, default=utc_now)

    uploader = relationship("User", back_populates="documents")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_docs_json = Column(Text, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=utc_now)

    user = relationship("User", back_populates="chat_histories")

