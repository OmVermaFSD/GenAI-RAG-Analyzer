from datetime import datetime, timedelta, timezone
from typing import Optional
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.config.settings import settings
from app.db.database import get_db
from app.models.db_models import User
from app.utils.exceptions import AuthenticationError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class AuthService:
    """Service handling user password hashing, JWT creation, and authentication."""

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)

    @classmethod
    def create_access_token(cls, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    @classmethod
    def authenticate_user(cls, db: Session, username: str, password: str) -> Optional[User]:
        user = db.query(User).filter(User.username == username).first()
        if not user or not cls.verify_password(password, user.hashed_password):
            return None
        return user

    @classmethod
    def register_user(cls, db: Session, username: str, password: str) -> User:
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        hashed_password = cls.get_password_hash(password)
        db_user = User(username=username, hashed_password=hashed_password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

def get_current_user(token: Optional[str] = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Optional[User]:
    """Dependency to retrieve authenticated user from JWT bearer token."""
    if not token:
        # If no token is provided, raise 401 AuthenticationError
        raise AuthenticationError("Authentication required. Please provide a valid Bearer token.")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token payload")
    except jwt.PyJWTError:
        raise AuthenticationError("Could not validate credentials")

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise AuthenticationError("User not found")
    return user
