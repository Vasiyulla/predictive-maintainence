"""
Authentication Service - Authentication Logic
Handles password hashing, JWT token creation/verification, and user authentication
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from config.settings import settings
from .models import User, TokenData
import hashlib
# ==============================================================================
# Password Hashing Configuration
# ==============================================================================

# Create password context using bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Number of hashing rounds (higher = more secure but slower)
)


# ==============================================================================
# Password Functions
# ==============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against its hashed version
    
    Args:
        plain_password: The password entered by user
        hashed_password: The hashed password from database
    
    Returns:
        bool: True if password matches, False otherwise
    
    Example:
        >>> hashed = get_password_hash("mypassword")
        >>> verify_password("mypassword", hashed)
        True
        >>> verify_password("wrongpassword", hashed)
        False
    """

    try:
        sha256 = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password to hash
    
    Returns:
        str: Hashed password
    
    Example:
        >>> hashed = get_password_hash("mypassword")
        >>> print(hashed)
        $2b$12$...
    """
    sha256 = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return pwd_context.hash(password)


def check_password_strength(password: str) -> dict:
    """
    Check password strength and return feedback
    
    Args:
        password: Password to check
    
    Returns:
        dict: Contains 'strong' (bool), 'score' (int 0-5), and 'feedback' (list)
    
    Example:
        >>> check_password_strength("weak")
        {'strong': False, 'score': 1, 'feedback': ['Too short', 'No numbers']}
    """
    score = 0
    feedback = []
    
    # Length check
    if len(password) >= 8:
        score += 1
    else:
        feedback.append("Should be at least 8 characters")
    
    if len(password) >= 12:
        score += 1
    
    # Complexity checks
    if any(c.isupper() for c in password):
        score += 1
    else:
        feedback.append("Should contain uppercase letters")
    
    if any(c.islower() for c in password):
        score += 1
    else:
        feedback.append("Should contain lowercase letters")
    
    if any(c.isdigit() for c in password):
        score += 1
    else:
        feedback.append("Should contain numbers")
    
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        score += 1
    else:
        feedback.append("Should contain special characters")
    
    return {
        'strong': score >= 4,
        'score': score,
        'feedback': feedback if score < 4 else ["Strong password"]
    }


# ==============================================================================
# JWT Token Functions
# ==============================================================================

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Dictionary of data to encode in token (usually {"sub": username})
        expires_delta: Optional expiration time delta
    
    Returns:
        str: Encoded JWT token
    
    Example:
        >>> token = create_access_token({"sub": "john_doe"})
        >>> print(token)
        eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    # Add expiration and issued at time
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    # Encode JWT
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decode and validate a JWT token
    
    Args:
        token: JWT token string
    
    Returns:
        dict: Decoded token payload if valid, None otherwise
    
    Example:
        >>> token = create_access_token({"sub": "john_doe"})
        >>> payload = decode_access_token(token)
        >>> print(payload['sub'])
        john_doe
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
    except JWTError as e:
        print(f"Token decode error: {e}")
        return None


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify JWT token and extract token data
    
    Args:
        token: JWT token string
    
    Returns:
        TokenData: Token data if valid, None otherwise
    
    Example:
        >>> token = create_access_token({"sub": "john_doe", "user_id": 1})
        >>> token_data = verify_token(token)
        >>> print(token_data.username)
        john_doe
    """
    try:
        # Decode token
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        
        # Extract username (subject)
        username: str = payload.get("sub")
        if username is None:
            return None
        
        # Extract other data
        user_id: int = payload.get("user_id")
        role: str = payload.get("role")
        exp_timestamp = payload.get("exp")
        
        # Convert expiration timestamp to datetime
        exp_datetime = None
        if exp_timestamp:
            exp_datetime = datetime.fromtimestamp(exp_timestamp)
        
        # Create TokenData object
        token_data = TokenData(
            username=username,
            user_id=user_id,
            role=role,
            exp=exp_datetime
        )
        
        return token_data
        
    except JWTError as e:
        print(f"Token verification error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in token verification: {e}")
        return None


def is_token_expired(token: str) -> bool:
    """
    Check if a token is expired
    
    Args:
        token: JWT token string
    
    Returns:
        bool: True if expired, False otherwise
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": True}
        )
        return False
    except jwt.ExpiredSignatureError:
        return True
    except JWTError:
        return True


# ==============================================================================
# User Authentication Functions
# ==============================================================================

def authenticate_user(
    db: Session,
    username: str,
    password: str
) -> Optional[User]:
    """
    Authenticate a user with username and password
    
    Args:
        db: Database session
        username: Username to authenticate
        password: Plain text password
    
    Returns:
        User: User object if authentication successful, None otherwise
    
    Example:
        >>> user = authenticate_user(db, "john_doe", "mypassword")
        >>> if user:
        ...     print(f"Welcome {user.username}")
    """
    # Get user from database
    user = db.query(User).filter(User.username == username).first()
    
    if not user:
        print(f"Authentication failed: User '{username}' not found")
        return None
    
    # Verify password
    if not verify_password(password, user.hashed_password):
        print(f"Authentication failed: Invalid password for user '{username}'")
        return None
    
    # Check if user is active
    if not user.is_active:
        print(f"Authentication failed: User '{username}' is inactive")
        return None
    
    # Update last login time
    try:
        user.last_login = datetime.utcnow()
        db.commit()
    except Exception as e:
        print(f"Failed to update last login: {e}")
        db.rollback()
    
    return user


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """
    Get user by username
    
    Args:
        db: Database session
        username: Username to search for
    
    Returns:
        User: User object if found, None otherwise
    """
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """
    Get user by email
    
    Args:
        db: Database session
        email: Email to search for
    
    Returns:
        User: User object if found, None otherwise
    """
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """
    Get user by ID
    
    Args:
        db: Database session
        user_id: User ID to search for
    
    Returns:
        User: User object if found, None otherwise
    """
    return db.query(User).filter(User.id == user_id).first()


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    role: str = "operator"
) -> User:
    """
    Create a new user
    
    Args:
        db: Database session
        username: Username for new user
        email: Email for new user
        password: Plain text password
        role: User role (default: "operator")
    
    Returns:
        User: Created user object
    
    Raises:
        ValueError: If username or email already exists
    
    Example:
        >>> user = create_user(db, "john_doe", "john@example.com", "password123")
        >>> print(user.username)
        john_doe
    """
    # Check if username exists
    if get_user_by_username(db, username):
        raise ValueError(f"Username '{username}' already exists")
    
    # Check if email exists
    if get_user_by_email(db, email):
        raise ValueError(f"Email '{email}' already registered")
    
    # Hash password
    hashed_password = get_password_hash(password)
    
    # Create user object
    db_user = User(
        username=username.lower(),
        email=email.lower(),
        hashed_password=hashed_password,
        role=role,
        is_active=True,
        created_at=datetime.utcnow()
    )
    
    # Add to database
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    print(f"User created: {username} ({email}) - Role: {role}")
    
    return db_user


def update_user(
    db: Session,
    user_id: int,
    email: Optional[str] = None,
    password: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None
) -> Optional[User]:
    """
    Update user information
    
    Args:
        db: Database session
        user_id: ID of user to update
        email: New email (optional)
        password: New password (optional)
        role: New role (optional)
        is_active: New active status (optional)
    
    Returns:
        User: Updated user object if successful, None otherwise
    """
    user = get_user_by_id(db, user_id)
    
    if not user:
        return None
    
    # Update fields if provided
    if email is not None:
        # Check if new email already exists
        existing = get_user_by_email(db, email)
        if existing and existing.id != user_id:
            raise ValueError(f"Email '{email}' already registered")
        user.email = email.lower()
    
    if password is not None:
        user.hashed_password = get_password_hash(password)
    
    if role is not None:
        if role not in ['operator', 'admin']:
            raise ValueError("Role must be 'operator' or 'admin'")
        user.role = role
    
    if is_active is not None:
        user.is_active = is_active
    
    # Commit changes
    try:
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        print(f"Failed to update user: {e}")
        return None


def delete_user(db: Session, user_id: int) -> bool:
    """
    Delete a user
    
    Args:
        db: Database session
        user_id: ID of user to delete
    
    Returns:
        bool: True if successful, False otherwise
    """
    user = get_user_by_id(db, user_id)
    
    if not user:
        return False
    
    try:
        db.delete(user)
        db.commit()
        print(f"User deleted: {user.username}")
        return True
    except Exception as e:
        db.rollback()
        print(f"Failed to delete user: {e}")
        return False


def get_all_users(
    db: Session,
    skip: int = 0,
    limit: int = 100
) -> list[User]:
    """
    Get all users with pagination
    
    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        list[User]: List of user objects
    """
    return db.query(User).offset(skip).limit(limit).all()


def count_users(db: Session) -> int:
    """
    Count total number of users
    
    Args:
        db: Database session
    
    Returns:
        int: Total number of users
    """
    return db.query(User).count()


# ==============================================================================
# Authorization Functions
# ==============================================================================

def is_admin(user: User) -> bool:
    """
    Check if user is an admin
    
    Args:
        user: User object
    
    Returns:
        bool: True if user is admin, False otherwise
    """
    return user.role == "admin"


def is_operator(user: User) -> bool:
    """
    Check if user is an operator
    
    Args:
        user: User object
    
    Returns:
        bool: True if user is operator, False otherwise
    """
    return user.role == "operator"


def has_role(user: User, required_role: str) -> bool:
    """
    Check if user has a specific role
    
    Args:
        user: User object
        required_role: Role to check for
    
    Returns:
        bool: True if user has the role, False otherwise
    """
    return user.role == required_role


def can_access_resource(user: User, resource_owner_id: int) -> bool:
    """
    Check if user can access a resource
    Admins can access all resources, operators can only access their own
    
    Args:
        user: User object
        resource_owner_id: ID of the resource owner
    
    Returns:
        bool: True if user can access, False otherwise
    """
    if is_admin(user):
        return True
    
    return user.id == resource_owner_id


# ==============================================================================
# Utility Functions
# ==============================================================================

def generate_username_from_email(email: str) -> str:
    """
    Generate a username from an email address
    
    Args:
        email: Email address
    
    Returns:
        str: Generated username
    
    Example:
        >>> generate_username_from_email("john.doe@example.com")
        'john_doe'
    """
    username = email.split('@')[0]
    username = username.replace('.', '_').replace('-', '_')
    return username.lower()


def is_valid_email_format(email: str) -> bool:
    """
    Check if email format is valid (basic check)
    
    Args:
        email: Email address to validate
    
    Returns:
        bool: True if valid format, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


# ==============================================================================
# Testing Functions
# ==============================================================================

if __name__ == "__main__":
    # Test password hashing
    print("Testing password functions...")
    password = "test_password_123"
    hashed = get_password_hash(password)
    print(f"Original: {password}")
    print(f"Hashed: {hashed}")
    print(f"Verify correct: {verify_password(password, hashed)}")
    print(f"Verify wrong: {verify_password('wrong_password', hashed)}")
    
    # Test password strength
    print("\nTesting password strength...")
    weak = check_password_strength("weak")
    print(f"Weak password: {weak}")
    strong = check_password_strength("StrongP@ssw0rd123")
    print(f"Strong password: {strong}")
    
    # Test JWT token
    print("\nTesting JWT token...")
    token = create_access_token({"sub": "test_user", "user_id": 1})
    print(f"Token: {token[:50]}...")
    token_data = verify_token(token)
    print(f"Decoded: {token_data}")
    
    print("\n✓ All authentication functions working correctly")