import os
from datetime import datetime, timedelta
import secrets
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import psycopg

from .database import get_db, get_user_by_username, get_user_by_api_key
from .schemas import TokenData
from .utils import setup_logger

# Setup logger
logger = setup_logger("auth")

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def generate_api_key():
    """Generate a secure API key for user authentication."""
    return secrets.token_urlsafe(32)


def authenticate_user(conn, username: str, password: str):
    user = get_user_by_username(username)
    if not user or not verify_password(password, user['password_hash']):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), conn = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        logger.error("JWT token validation failed")
        raise credentials_exception
    
    user = get_user_by_username(username)
    if user is None:
        logger.error(f"User {token_data.username} not found in database")
        raise credentials_exception
    return user


async def get_current_active_user(current_user = Depends(get_current_user)):
    if not current_user['is_active']:
        logger.warning(f"Inactive user {current_user['username']} attempted to access API")
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def verify_api_key(api_key: str):
    """Verify API key and return user if valid."""
    user = get_user_by_api_key(api_key)
    if not user or not user['is_active']:
        return None 