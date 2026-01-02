"""
Authentication Service - Database Models
Defines User model and Pydantic schemas for API validation
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from .database import Base

# ==============================================================================
# SQLAlchemy ORM Models (Database Tables)
# ==============================================================================

class User(Base):
    """
    User database model
    Represents a user in the system with authentication details
    """
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}  # ✅ ADD THIS
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # User credentials
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # User role and status
    role = Column(String(20), default="operator", nullable=False)  # operator, admin
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)  # ✅ FIX

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', role='{self.role}')>"
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }


# ==============================================================================
# Pydantic Schemas (API Request/Response Models)
# ==============================================================================

class UserBase(BaseModel):
    """Base user schema with common fields"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """Validate username is alphanumeric with underscores"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v.lower()
    
    @validator('email')
    def email_lowercase(cls, v):
        """Convert email to lowercase"""
        return v.lower()


class UserCreate(UserBase):
    """Schema for user registration/creation"""
    password: str = Field(
        ..., 
        min_length=6, 
        max_length=100,
        description="Password (minimum 6 characters)"
    )
    role: str = Field(
        default="operator",
        description="User role (operator or admin)"
    )
    
    @validator('role')
    def validate_role(cls, v):
        """Validate role is either operator or admin"""
        if v not in ['operator', 'admin']:
            raise ValueError('Role must be either "operator" or "admin"')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength"""
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        if v.isalnum() and v.isalpha():
            # Encourage but don't require special characters
            pass
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "email": "john@example.com",
                "password": "secure_password_123",
                "role": "operator"
            }
        }


class UserUpdate(BaseModel):
    """Schema for updating user information"""
    email: Optional[EmailStr] = Field(None, description="New email address")
    password: Optional[str] = Field(None, min_length=6, description="New password")
    role: Optional[str] = Field(None, description="New role")
    is_active: Optional[bool] = Field(None, description="Active status")
    
    @validator('role')
    def validate_role(cls, v):
        """Validate role if provided"""
        if v is not None and v not in ['operator', 'admin']:
            raise ValueError('Role must be either "operator" or "admin"')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "newemail@example.com",
                "role": "admin"
            }
        }


class UserLogin(BaseModel):
    """Schema for user login"""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")
    
    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "password": "secure_password_123"
            }
        }


class UserResponse(BaseModel):
    """Schema for user data response (excludes password)"""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True  # Pydantic v2
        orm_mode = True  # Pydantic v1 compatibility
        schema_extra = {
            "example": {
                "id": 1,
                "username": "john_doe",
                "email": "john@example.com",
                "role": "operator",
                "is_active": True,
                "created_at": "2024-01-01T00:00:00",
                "last_login": "2024-01-02T10:30:00"
            }
        }


class Token(BaseModel):
    """Schema for JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: UserResponse = Field(..., description="User information")
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "user": {
                    "id": 1,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "role": "operator",
                    "is_active": True,
                    "created_at": "2024-01-01T00:00:00",
                    "last_login": "2024-01-02T10:30:00"
                }
            }
        }


class TokenData(BaseModel):
    """Schema for token payload (JWT claims)"""
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "username": "john_doe",
                "user_id": 1,
                "role": "operator",
                "exp": "2024-01-03T10:30:00"
            }
        }


class TokenVerifyRequest(BaseModel):
    """Schema for token verification request"""
    token: str = Field(..., description="JWT token to verify")
    
    class Config:
        schema_extra = {
            "example": {
                "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenVerifyResponse(BaseModel):
    """Schema for token verification response"""
    valid: bool = Field(..., description="Whether token is valid")
    user: Optional[UserResponse] = Field(None, description="User information if valid")
    message: Optional[str] = Field(None, description="Error message if invalid")
    
    class Config:
        schema_extra = {
            "example": {
                "valid": True,
                "user": {
                    "id": 1,
                    "username": "john_doe",
                    "email": "john@example.com",
                    "role": "operator",
                    "is_active": True,
                    "created_at": "2024-01-01T00:00:00",
                    "last_login": "2024-01-02T10:30:00"
                }
            }
        }


class UserListResponse(BaseModel):
    """Schema for listing multiple users"""
    total: int = Field(..., description="Total number of users")
    users: list[UserResponse] = Field(..., description="List of users")
    
    class Config:
        schema_extra = {
            "example": {
                "total": 2,
                "users": [
                    {
                        "id": 1,
                        "username": "admin",
                        "email": "admin@example.com",
                        "role": "admin",
                        "is_active": True,
                        "created_at": "2024-01-01T00:00:00",
                        "last_login": "2024-01-02T10:30:00"
                    },
                    {
                        "id": 2,
                        "username": "operator",
                        "email": "operator@example.com",
                        "role": "operator",
                        "is_active": True,
                        "created_at": "2024-01-01T00:00:00",
                        "last_login": None
                    }
                ]
            }
        }


class MessageResponse(BaseModel):
    """Generic message response"""
    message: str = Field(..., description="Response message")
    detail: Optional[str] = Field(None, description="Additional details")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Operation successful",
                "detail": "User updated successfully"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Username already exists"
            }
        }