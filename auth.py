"""
Authentication Module for Smart DentalOps
Handles user login, registration, JWT tokens, and role-based access control
"""

from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, EmailStr
import hashlib
import os
from jose import JWTError, jwt

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Simple password hashing using hashlib
def hash_password_simple(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password_simple(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return hash_password_simple(plain_password) == hashed_password

# ============================================================================
# DATA MODELS
# ============================================================================

class UserBase(BaseModel):
    """Base user model"""
    email: EmailStr
    full_name: str
    role: str  # "admin", "dentist", "staff", "patient"

class UserCreate(UserBase):
    """User creation model"""
    password: str

class UserLogin(BaseModel):
    """User login model"""
    email: EmailStr
    password: str

class User(UserBase):
    """User model (response)"""
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str
    user: User

class TokenData(BaseModel):
    """Token data model"""
    email: Optional[str] = None
    role: Optional[str] = None

# ============================================================================
# MOCK DATABASE (Replace with real database in production)
# ============================================================================

# In-memory user storage (for demo purposes)
# Initialize lazily to avoid bcrypt issues at import time
USERS_DB = {}

# ============================================================================
# PASSWORD HASHING
# ============================================================================

def _initialize_demo_users():
    """Initialize demo users with hashed passwords"""
    global USERS_DB
    if not USERS_DB:  # Only initialize once
        USERS_DB = {
            "admin@dentalops.com": {
                "id": 1,
                "email": "admin@dentalops.com",
                "full_name": "Admin User",
                "hashed_password": hash_password_simple("admin123"),
                "role": "admin",
                "is_active": True,
                "created_at": datetime.now()
            },
            "dentist@dentalops.com": {
                "id": 2,
                "email": "dentist@dentalops.com",
                "full_name": "Dr. Smith",
                "hashed_password": hash_password_simple("dentist123"),
                "role": "dentist",
                "is_active": True,
                "created_at": datetime.now()
            },
            "staff@dentalops.com": {
                "id": 3,
                "email": "staff@dentalops.com",
                "full_name": "Staff Member",
                "hashed_password": hash_password_simple("staff123"),
                "role": "staff",
                "is_active": True,
                "created_at": datetime.now()
            }
        }

def hash_password(password: str) -> str:
    """Hash a password"""
    return hash_password_simple(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return verify_password_simple(plain_password, hashed_password)

# ============================================================================
# JWT TOKEN FUNCTIONS
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt

def decode_token(token: str) -> Optional[TokenData]:
    """Decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        
        if email is None:
            return None
        
        return TokenData(email=email, role=role)
    except JWTError:
        return None

# ============================================================================
# USER MANAGEMENT
# ============================================================================

def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email"""
    _initialize_demo_users()  # Ensure demo users are initialized
    return USERS_DB.get(email)

def create_user(user: UserCreate) -> User:
    """Create a new user"""
    _initialize_demo_users()  # Ensure demo users are initialized
    
    if user.email in USERS_DB:
        raise ValueError("User already exists")
    
    user_id = len(USERS_DB) + 1
    hashed_password = hash_password(user.password)
    
    db_user = {
        "id": user_id,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "role": user.role,
        "is_active": True,
        "created_at": datetime.now()
    }
    
    USERS_DB[user.email] = db_user
    
    return User(**db_user)

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user with email and password"""
    _initialize_demo_users()  # Ensure demo users are initialized
    
    user = get_user_by_email(email)
    
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    if not user["is_active"]:
        return None
    
    return user

# ============================================================================
# ROLE-BASED ACCESS CONTROL
# ============================================================================

ROLE_PERMISSIONS = {
    "admin": ["read", "write", "delete", "manage_users"],
    "dentist": ["read", "write", "manage_appointments"],
    "staff": ["read", "write"],
    "patient": ["read"]
}

def has_permission(role: str, permission: str) -> bool:
    """Check if role has permission"""
    permissions = ROLE_PERMISSIONS.get(role, [])
    return permission in permissions

def check_role(required_role: str, user_role: str) -> bool:
    """Check if user has required role"""
    role_hierarchy = {
        "admin": 4,
        "dentist": 3,
        "staff": 2,
        "patient": 1
    }
    
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    
    return user_level >= required_level

# ============================================================================
# DEMO CREDENTIALS
# ============================================================================

DEMO_USERS = {
    "admin": {
        "email": "admin@dentalops.com",
        "password": "admin123",
        "role": "admin"
    },
    "dentist": {
        "email": "dentist@dentalops.com",
        "password": "dentist123",
        "role": "dentist"
    },
    "staff": {
        "email": "staff@dentalops.com",
        "password": "staff123",
        "role": "staff"
    }
}
