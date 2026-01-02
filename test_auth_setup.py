#!/usr/bin/env python3
"""
Test script to verify auth models and auth logic are working correctly
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    try:
        from services.auth_service.models import (
            User, UserCreate, UserLogin, UserResponse, 
            Token, TokenData
        )
        print("✓ Models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from services.auth_service.auth import (
            get_password_hash, verify_password,
            create_access_token, verify_token,
            authenticate_user, create_user
        )
        print("✓ Auth functions imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import auth functions: {e}")
        return False
    
    try:
        from services.auth_service.database import (
            engine, Base, SessionLocal
        )
        print("✓ Database components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import database: {e}")
        return False
    
    return True


def test_password_functions():
    """Test password hashing and verification"""
    print("\n" + "="*60)
    print("TESTING PASSWORD FUNCTIONS")
    print("="*60)
    
    from services.auth_service.auth import (
        get_password_hash, verify_password, check_password_strength
    )
    
    # Test hashing
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    print(f"\nOriginal password: {password}")
    print(f"Hashed password: {hashed[:50]}...")
    
    # Test verification
    if verify_password(password, hashed):
        print("✓ Password verification successful (correct password)")
    else:
        print("✗ Password verification failed (should have succeeded)")
        return False
    
    if not verify_password("wrong_password", hashed):
        print("✓ Password verification successful (rejected wrong password)")
    else:
        print("✗ Password verification failed (should have rejected)")
        return False
    
    # Test password strength
    weak = check_password_strength("weak")
    strong = check_password_strength("StrongP@ssw0rd123")
    
    print(f"\nWeak password check: Score {weak['score']}/6")
    print(f"Strong password check: Score {strong['score']}/6")
    
    if strong['strong']:
        print("✓ Password strength check working")
    else:
        print("⚠ Password strength check might need adjustment")
    
    return True


def test_jwt_functions():
    """Test JWT token creation and verification"""
    print("\n" + "="*60)
    print("TESTING JWT FUNCTIONS")
    print("="*60)
    
    from services.auth_service.auth import (
        create_access_token, verify_token, decode_access_token
    )
    
    # Create token
    test_data = {
        "sub": "test_user",
        "user_id": 1,
        "role": "admin"
    }
    
    token = create_access_token(test_data)
    print(f"\nCreated token: {token[:50]}...")
    
    # Verify token
    token_data = verify_token(token)
    
    if token_data:
        print(f"✓ Token verified successfully")
        print(f"  Username: {token_data.username}")
        print(f"  User ID: {token_data.user_id}")
        print(f"  Role: {token_data.role}")
    else:
        print("✗ Token verification failed")
        return False
    
    # Decode token
    payload = decode_access_token(token)
    
    if payload and payload.get("sub") == "test_user":
        print("✓ Token decoding successful")
    else:
        print("✗ Token decoding failed")
        return False
    
    return True


def test_database_models():
    """Test database models"""
    print("\n" + "="*60)
    print("TESTING DATABASE MODELS")
    print("="*60)
    
    from services.auth_service.database import Base, engine, SessionLocal
    from services.auth_service.models import User
    from services.auth_service.auth import get_password_hash
    from datetime import datetime
    
    # Create tables
    try:
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created")
    except Exception as e:
        print(f"✗ Failed to create tables: {e}")
        return False
    
    # Test creating a user
    db = SessionLocal()
    
    try:
        # Check if test user already exists
        existing = db.query(User).filter(User.username == "test_user_temp").first()
        if existing:
            db.delete(existing)
            db.commit()
        
        # Create test user
        test_user = User(
            username="test_user_temp",
            email="test@example.com",
            hashed_password=get_password_hash("password123"),
            role="operator",
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow()
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print(f"✓ Test user created (ID: {test_user.id})")
        
        # Test user retrieval
        retrieved = db.query(User).filter(User.username == "test_user_temp").first()
        
        if retrieved:
            print(f"✓ User retrieved from database")
            print(f"  Username: {retrieved.username}")
            print(f"  Email: {retrieved.email}")
            print(f"  Role: {retrieved.role}")
        else:
            print("✗ Failed to retrieve user")
            return False
        
        # Clean up
        db.delete(test_user)
        db.commit()
        print("✓ Test user cleaned up")
        
    except Exception as e:
        print(f"✗ Database operation failed: {e}")
        db.rollback()
        return False
    finally:
        db.close()
    
    return True


def test_user_operations():
    """Test user CRUD operations"""
    print("\n" + "="*60)
    print("TESTING USER OPERATIONS")
    print("="*60)
    
    from services.auth_service.database import SessionLocal
    from services.auth_service.auth import (
        create_user, get_user_by_username, authenticate_user,
        update_user, delete_user
    )
    
    db = SessionLocal()
    
    try:
        # Clean up any existing test user
        existing = get_user_by_username(db, "test_operations")
        if existing:
            delete_user(db, existing.id)
        
        # Test create user
        print("\n1. Testing create_user...")
        user = create_user(
            db=db,
            username="test_operations",
            email="testops@example.com",
            password="password123",
            role="operator"
        )
        print(f"✓ User created: {user.username}")
        
        # Test get user
        print("\n2. Testing get_user_by_username...")
        retrieved = get_user_by_username(db, "test_operations")
        if retrieved and retrieved.username == "test_operations":
            print(f"✓ User retrieved: {retrieved.username}")
        else:
            print("✗ Failed to retrieve user")
            return False
        
        # Test authenticate
        print("\n3. Testing authenticate_user...")
        authenticated = authenticate_user(db, "test_operations", "password123")
        if authenticated:
            print(f"✓ User authenticated: {authenticated.username}")
        else:
            print("✗ Authentication failed")
            return False
        
        # Test wrong password
        print("\n4. Testing authentication with wrong password...")
        wrong_auth = authenticate_user(db, "test_operations", "wrong_password")
        if not wrong_auth:
            print("✓ Correctly rejected wrong password")
        else:
            print("✗ Should have rejected wrong password")
            return False
        
        # Test update user
        print("\n5. Testing update_user...")
        updated = update_user(
            db=db,
            user_id=user.id,
            role="admin"
        )
        if updated and updated.role == "admin":
            print(f"✓ User updated: role changed to {updated.role}")
        else:
            print("✗ Failed to update user")
            return False
        
        # Test delete user
        print("\n6. Testing delete_user...")
        deleted = delete_user(db, user.id)
        if deleted:
            print("✓ User deleted successfully")
        else:
            print("✗ Failed to delete user")
            return False
        
        # Verify deletion
        check = get_user_by_username(db, "test_operations")
        if not check:
            print("✓ User deletion verified")
        else:
            print("✗ User still exists after deletion")
            return False
        
    except Exception as e:
        print(f"✗ User operation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()
    
    return True


def test_pydantic_schemas():
    """Test Pydantic schemas"""
    print("\n" + "="*60)
    print("TESTING PYDANTIC SCHEMAS")
    print("="*60)
    
    from services.auth_service.models import (
        UserCreate, UserLogin, UserResponse, Token
    )
    
    # Test UserCreate
    print("\n1. Testing UserCreate schema...")
    try:
        user_create = UserCreate(
            username="john_doe",
            email="john@example.com",
            password="password123",
            role="operator"
        )
        print(f"✓ UserCreate valid: {user_create.username}")
    except Exception as e:
        print(f"✗ UserCreate validation failed: {e}")
        return False
    
    # Test invalid email
    print("\n2. Testing email validation...")
    try:
        invalid_user = UserCreate(
            username="test",
            email="invalid_email",
            password="password123"
        )
        print("✗ Should have rejected invalid email")
        return False
    except Exception:
        print("✓ Correctly rejected invalid email")
    
    # Test UserLogin
    print("\n3. Testing UserLogin schema...")
    try:
        user_login = UserLogin(
            username="john_doe",
            password="password123"
        )
        print(f"✓ UserLogin valid: {user_login.username}")
    except Exception as e:
        print(f"✗ UserLogin validation failed: {e}")
        return False
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("PREDICTXAI - AUTHENTICATION MODULE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Import Test", test_imports),
        ("Password Functions", test_password_functions),
        ("JWT Functions", test_jwt_functions),
        ("Database Models", test_database_models),
        ("User Operations", test_user_operations),
        ("Pydantic Schemas", test_pydantic_schemas)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nAuth system is working correctly!")
        print("You can now:")
        print("  1. Run: python init_database.py --all")
        print("  2. Start services with: ./quickstart.sh")
        print("="*70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())