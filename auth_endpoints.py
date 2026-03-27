"""
Authentication Endpoints for Smart DentalOps
Add these endpoints to main.py
"""

from fastapi import APIRouter, HTTPException, Depends, status, Header
from datetime import timedelta
from typing import Optional
from auth import (
    UserCreate, UserLogin, User, Token, TokenData,
    create_user, authenticate_user, create_access_token,
    decode_token, get_user_by_email, ACCESS_TOKEN_EXPIRE_MINUTES,
    DEMO_USERS
)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Helper function to extract token from Authorization header
def get_token_from_header(authorization: Optional[str] = Header(None)) -> str:
    """Extract bearer token from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return parts[1]

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@router.post("/register", response_model=User)
async def register(user: UserCreate):
    """
    Register a new user
    
    Args:
        user: UserCreate object with email, full_name, password, role
    
    Returns:
        User object
    """
    try:
        new_user = create_user(user)
        return new_user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """
    Login user and return JWT token
    
    Args:
        credentials: UserLogin object with email and password
    
    Returns:
        Token object with access_token, token_type, and user
    """
    user = authenticate_user(credentials.email, credentials.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    # Return token and user info
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=User(**user)
    )

@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """
    Logout user (invalidate token)
    
    Note: In production, add token to blacklist
    """
    token = get_token_from_header(authorization)
    return {"message": "Logged out successfully"}

@router.get("/me", response_model=User)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Get current logged-in user
    
    Args:
        authorization: Bearer token in Authorization header
    
    Returns:
        Current user object
    """
    token = get_token_from_header(authorization)
    token_data = decode_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user = get_user_by_email(token_data.email)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return User(**user)

@router.post("/refresh-token", response_model=Token)
async def refresh_token(authorization: Optional[str] = Header(None)):
    """
    Refresh access token
    
    Args:
        authorization: Current bearer token in Authorization header
    
    Returns:
        New token object
    """
    token = get_token_from_header(authorization)
    token_data = decode_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    user = get_user_by_email(token_data.email)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]},
        expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=User(**user)
    )

@router.get("/demo-credentials")
async def get_demo_credentials():
    """
    Get demo credentials for testing
    
    Returns:
        Dictionary with demo user credentials
    """
    return {
        "message": "Demo credentials for testing",
        "users": DEMO_USERS
    }

# ============================================================================
# PROTECTED ENDPOINTS EXAMPLE
# ============================================================================

@router.get("/protected-example")
async def protected_endpoint(authorization: Optional[str] = Header(None)):
    """
    Example of a protected endpoint
    
    Args:
        authorization: Bearer token in Authorization header
    
    Returns:
        Protected data
    """
    token = get_token_from_header(authorization)
    token_data = decode_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return {
        "message": f"Hello {token_data.email}",
        "role": token_data.role,
        "data": "This is protected data"
    }
