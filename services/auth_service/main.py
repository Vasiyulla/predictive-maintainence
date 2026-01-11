"""
Authentication Service API
Microservice for user authentication and authorization
"""
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from datetime import timedelta

from config.settings import settings
from .database import get_db, init_db
from .models import UserCreate, UserLogin, UserResponse, Token, Machine, MachineCreate, MachineResponse
from .auth import (
    authenticate_user,
    create_access_token,
    get_user_by_username,
    get_user_by_email,
    create_user,
    verify_token
)

# Initialize FastAPI app
app = FastAPI(
    title=f"{settings.APP_NAME} - Auth Service",
    version=settings.APP_VERSION,
    description="Authentication and Authorization Service"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "service": "auth_service",
        "status": "healthy",
        "version": settings.APP_VERSION
    }

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    # Check if username exists
    if get_user_by_username(db, user.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    if get_user_by_email(db, user.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    db_user = create_user(
        db=db,
        username=user.username,
        email=user.email,
        password=user.password,
        role=user.role
    )
    
    return db_user

@app.post("/login", response_model=Token)
async def login(user_login: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token"""
    user = authenticate_user(db, user_login.username, user_login.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@app.post("/verify")
async def verify_user_token(token: str, db: Session = Depends(get_db)):
    """Verify JWT token and return user info"""
    token_data = verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = get_user_by_username(db, token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return {
        "valid": True,
        "user": UserResponse.from_orm(user)
    }
@app.post("/machines", response_model=MachineResponse, status_code=201)
async def create_machine(machine: MachineCreate, db: Session = Depends(get_db)):
    """Add a new machine"""
    existing = db.query(Machine).filter(Machine.name == machine.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Machine already exists")
    
    new_machine = Machine(
        name=machine.name,
        type=machine.type,
        location=machine.location,
        features=machine.features,
        settings=machine.settings,
        is_active=True
    )
    db.add(new_machine)
    db.commit()
    db.refresh(new_machine)
    return new_machine

@app.get("/machines", response_model=List[MachineResponse])
async def get_machines(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get all active machines"""
    return db.query(Machine).filter(Machine.is_active == True).offset(skip).limit(limit).all()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.AUTH_SERVICE_HOST,
        port=settings.AUTH_SERVICE_PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
# ... [Existing Auth Routes] ...

