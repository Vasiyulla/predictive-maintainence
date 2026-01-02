#!/usr/bin/env python3
"""
Database Initialization Script
Creates database tables and optional test users
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.auth_service.database import engine, Base, SessionLocal
from services.auth_service.models import User
from services.auth_service.auth import get_password_hash
from datetime import datetime
import bcrypt
def init_database():
    """Initialize database tables"""
    print("="*60)
    print("DATABASE INITIALIZATION")
    print("="*60)
    
    # Create tables
    print("\nCreating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created successfully")
    
    # Check database location
    db_path = project_root / "database" / "predictxai.db"
    print(f"\n✓ Database location: {db_path}")
    print(f"✓ Database size: {db_path.stat().st_size if db_path.exists() else 0} bytes")
    
    return True

def create_test_users():
    """Create default test users"""
    print("\n" + "="*60)
    print("CREATING TEST USERS")
    print("="*60)
    
    db = SessionLocal()
    
    try:
        # Check if users already exist
        existing_users = db.query(User).count()
        
        if existing_users > 0:
            print(f"\n⚠ Database already contains {existing_users} user(s)")
            response = input("Create additional users anyway? (y/n): ")
            if response.lower() != 'y':
                print("Skipping user creation.")
                return
        
        # Default test users
        test_users = [
            {
                "username": "admin",
                "email": "admin@predictxai.com",
                "password": "admin123",
                "role": "admin"
            },
            {
                "username": "operator",
                "email": "operator@predictxai.com",
                "password": "operator123",
                "role": "operator"
            }
        ]
        
        created_count = 0
        
        for user_data in test_users:
            # Check if user already exists
            existing = db.query(User).filter(
                User.username == user_data["username"]
            ).first()
            
            if existing:
                print(f"\n⚠ User '{user_data['username']}' already exists, skipping...")
                continue
            
            # Create user
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                hashed_password=get_password_hash(user_data["password"]),
                role=user_data["role"],
                is_active=True,
                created_at=datetime.utcnow()
            )
            
            db.add(user)
            db.commit()
            db.refresh(user)
            
            print(f"\n✓ Created user: {user.username}")
            print(f"  Email: {user.email}")
            print(f"  Role: {user.role}")
            print(f"  Password: {user_data['password']}")
            
            created_count += 1
        
        if created_count > 0:
            print(f"\n✓ Successfully created {created_count} user(s)")
        else:
            print("\n✓ No new users created")
            
    except Exception as e:
        print(f"\n✗ Error creating users: {e}")
        db.rollback()
    finally:
        db.close()

def verify_database():
    """Verify database setup"""
    print("\n" + "="*60)
    print("DATABASE VERIFICATION")
    print("="*60)
    
    db = SessionLocal()
    
    try:
        # Count users
        user_count = db.query(User).count()
        print(f"\n✓ Total users in database: {user_count}")
        
        # List all users
        if user_count > 0:
            print("\nRegistered users:")
            users = db.query(User).all()
            for user in users:
                print(f"  • {user.username} ({user.email}) - Role: {user.role}")
        
        # Test database operations
        print("\n✓ Database read operations working")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Database verification failed: {e}")
        return False
    finally:
        db.close()

def reset_database():
    """Drop all tables and recreate (CAUTION: Deletes all data)"""
    print("\n" + "="*60)
    print("⚠  WARNING: DATABASE RESET")
    print("="*60)
    print("\nThis will DELETE ALL DATA in the database!")
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Database reset cancelled.")
        return False
    
    print("\nDropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✓ All tables dropped")
    
    print("\nRecreating tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables recreated")
    
    return True

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PredictXAI Database Management')
    parser.add_argument(
        '--init',
        action='store_true',
        help='Initialize database tables'
    )
    parser.add_argument(
        '--create-users',
        action='store_true',
        help='Create test users'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify database setup'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database (CAUTION: Deletes all data)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Initialize database and create test users'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        print("="*60)
        print("PREDICTXAI - DATABASE MANAGEMENT")
        print("="*60)
        print("\nUsage:")
        print("  python init_database.py --init           # Initialize tables")
        print("  python init_database.py --create-users   # Create test users")
        print("  python init_database.py --verify         # Verify database")
        print("  python init_database.py --all            # Init + create users")
        print("  python init_database.py --reset          # Reset database")
        print("\nQuick start:")
        print("  python init_database.py --all")
        print("="*60)
        return
    
    try:
        if args.reset:
            if reset_database():
                print("\n✓ Database reset complete")
        
        if args.init or args.all:
            if init_database():
                print("\n✓ Database initialization complete")
        
        if args.create_users or args.all:
            create_test_users()
        
        if args.verify or args.all:
            verify_database()
        
        print("\n" + "="*60)
        print("DATABASE SETUP COMPLETE")
        print("="*60)
        print("\nDefault test credentials:")
        print("  Admin user:")
        print("    Username: admin")
        print("    Password: admin123")
        print("\n  Operator user:")
        print("    Username: operator")
        print("    Password: operator123")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()