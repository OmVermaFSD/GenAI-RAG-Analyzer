from typing import Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Natural language question to query against ingested documents",
        examples=["What is the main objective of this document?"]
    )
    model: Optional[str] = Field(
        None,
        description="Optional Ollama model override (llama3, mistral, deepseek)",
        examples=["llama3"]
    )

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, description="User username")
    password: str = Field(..., min_length=4, description="User password")

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, description="Desired username")
    password: str = Field(..., min_length=4, description="Password")
