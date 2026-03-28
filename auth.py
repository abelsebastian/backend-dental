"""
Authentication Module - SQLite backed
"""

from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, EmailStr
import hashlib
import os
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from database import UserDB, get_db, init_db

SECRET_KEY = os.getenv("SECRET_KEY", "dentalops-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ── Password helpers ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hash_password(plain) == hashed


# ── Pydantic models ───────────────────────────────────────────────────────────

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: str

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[TokenData]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        role = payload.get("role")
        if email is None:
            return None
        return TokenData(email=email, role=role)
    except JWTError:
        return None


# ── DB helpers ────────────────────────────────────────────────────────────────

def _user_to_dict(u: UserDB) -> dict:
    return {
        "id": u.id,
        "email": u.email,
        "full_name": u.full_name,
        "hashed_password": u.hashed_password,
        "role": u.role,
        "is_active": u.is_active,
        "created_at": u.created_at,
    }

def get_user_by_email(email: str, db: Session) -> Optional[dict]:
    u = db.query(UserDB).filter(UserDB.email == email).first()
    return _user_to_dict(u) if u else None

def create_user(user: UserCreate, db: Session) -> User:
    existing = db.query(UserDB).filter(UserDB.email == user.email).first()
    if existing:
        raise ValueError("User already exists")
    db_user = UserDB(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hash_password(user.password),
        role=user.role,
        is_active=True,
        created_at=datetime.now(),
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return User(**_user_to_dict(db_user))

def authenticate_user(email: str, password: str, db: Session) -> Optional[dict]:
    user = get_user_by_email(email, db)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if not user["is_active"]:
        return None
    return user


# ── Seed demo users on startup ────────────────────────────────────────────────

def seed_demo_users(db: Session):
    """Insert demo users if they don't exist yet"""
    demos = [
        ("admin@dentalops.com", "Admin User", "admin123", "admin"),
        ("dentist@dentalops.com", "Dr. Smith", "dentist123", "dentist"),
        ("staff@dentalops.com", "Staff Member", "staff123", "staff"),
    ]
    for email, name, password, role in demos:
        if not db.query(UserDB).filter(UserDB.email == email).first():
            db.add(UserDB(
                email=email,
                full_name=name,
                hashed_password=hash_password(password),
                role=role,
                is_active=True,
                created_at=datetime.now(),
            ))
    db.commit()


# ── RBAC ──────────────────────────────────────────────────────────────────────

ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users"],
    "dentist": ["read", "write", "manage_appointments"],
    "staff": ["read", "write"],
    "patient": ["read"],
}

def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, [])


# ── Demo credentials (for /auth/demo-credentials endpoint) ───────────────────

DEMO_USERS = {
    "admin":   {"email": "admin@dentalops.com",   "password": "admin123",   "role": "admin"},
    "dentist": {"email": "dentist@dentalops.com", "password": "dentist123", "role": "dentist"},
    "staff":   {"email": "staff@dentalops.com",   "password": "staff123",   "role": "staff"},
}
