from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.utils.logger import logger

class RAGException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class DocumentIngestionError(RAGException):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=status.HTTP_400_BAD_REQUEST)

class VectorStoreError(RAGException):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

class LLMQueryError(RAGException):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuthenticationError(RAGException):
    def __init__(self, message: str = "Invalid credentials or token"):
        super().__init__(message=message, status_code=status.HTTP_401_UNAUTHORIZED)

async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
    logger.error(f"RAGException handling path '{request.url.path}': {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "detail": exc.message,
            "status_code": exc.status_code
        }
    )

