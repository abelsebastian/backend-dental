"""
Authentication Endpoints - SQLite backed
"""

from fastapi import APIRouter, HTTPException, Depends, status, Header
from datetime import timedelta
from typing import Optional
from sqlalchemy.orm import Session

from database import get_db
from auth import (
    UserCreate, UserLogin, User, Token, TokenData,
    create_user, authenticate_user, create_access_token,
    decode_token, get_user_by_email,
    ACCESS_TOKEN_EXPIRE_MINUTES, DEMO_USERS
)

router = APIRouter(prefix="/auth", tags=["authentication"])


def get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    return parts[1]


# ── Register ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=User)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        return create_user(user, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")


# ── Login ─────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    user = authenticate_user(credentials.email, credentials.password, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return Token(access_token=access_token, token_type="bearer", user=User(**user))


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    get_token(authorization)
    return {"message": "Logged out successfully"}


# ── Get current user ──────────────────────────────────────────────────────────

@router.get("/me", response_model=User)
async def get_current_user(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    token = get_token(authorization)
    token_data = decode_token(token)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_email(token_data.email, db)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user)


# ── Refresh token ─────────────────────────────────────────────────────────────

@router.post("/refresh-token", response_model=Token)
async def refresh_token(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    token = get_token(authorization)
    token_data = decode_token(token)
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_email(token_data.email, db)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    access_token = create_access_token(data={"sub": user["email"], "role": user["role"]})
    return Token(access_token=access_token, token_type="bearer", user=User(**user))


# ── Demo credentials ──────────────────────────────────────────────────────────

@router.get("/demo-credentials")
async def get_demo_credentials():
    return {"message": "Demo credentials for testing", "users": DEMO_USERS}


# ── List all users (admin only) ───────────────────────────────────────────────

@router.get("/users")
async def list_users(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    token = get_token(authorization)
    token_data = decode_token(token)
    if not token_data or token_data.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    from database import UserDB
    users = db.query(UserDB).all()
    return [{"id": u.id, "email": u.email, "full_name": u.full_name, "role": u.role, "is_active": u.is_active, "created_at": u.created_at} for u in users]
