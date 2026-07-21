"""JWT auth: password hashing, token creation, current-user dependency."""
import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from database import User, get_db

load_dotenv()

SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGO = "HS256"
EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(p: str) -> str:
    return pwd.hash(p)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd.verify(plain, hashed)


def create_token(username: str) -> str:
    payload = {"sub": username,
               "exp": datetime.utcnow() + timedelta(minutes=EXPIRE_MIN)}
    return jwt.encode(payload, SECRET, algorithm=ALGO)


def get_current_user(token: str = Depends(oauth2),
                     db: Session = Depends(get_db)) -> User:
    err = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token",
                        headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        username = payload.get("sub")
        if not username:
            raise err
    except JWTError:
        raise err
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise err
    return user
