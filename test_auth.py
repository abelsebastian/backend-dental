"""
Test Suite for Authentication System (Phase 4)
Tests all authentication endpoints and functionality
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header(text):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}ℹ️  {text}{RESET}")

# ============================================================================
# TEST 1: Get Demo Credentials
# ============================================================================
def test_demo_credentials():
    print_header("TEST 1: Get Demo Credentials")
    
    try:
        response = requests.get(f"{BASE_URL}/auth/demo-credentials")
        
        if response.status_code == 200:
            data = response.json()
            print_success("Demo credentials retrieved successfully")
            print(f"Available demo users: {len(data.get('users', []))}")
            
            for user in data.get('users', []):
                print(f"  • {user['email']} (Role: {user['role']})")
            
            return data.get('users', [])
        else:
            print_error(f"Failed to get demo credentials: {response.status_code}")
            return []
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return []

# ============================================================================
# TEST 2: User Registration
# ============================================================================
def test_registration():
    print_header("TEST 2: User Registration")
    
    test_user = {
        "email": f"testuser_{datetime.now().timestamp()}@example.com",
        "full_name": "Test User",
        "password": "TestPassword123",
        "role": "patient"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/register",
            json=test_user
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("User registration successful")
            print(f"  Email: {data['email']}")
            print(f"  Name: {data['full_name']}")
            print(f"  Role: {data['role']}")
            return test_user
        else:
            print_error(f"Registration failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return None

# ============================================================================
# TEST 3: User Login
# ============================================================================
def test_login(demo_users):
    print_header("TEST 3: User Login")
    
    if not demo_users:
        print_error("No demo users available")
        return None
    
    # Use first demo user
    login_user = demo_users[0]
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={
                "email": login_user["email"],
                "password": login_user["password"]
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get('access_token')
            user = data.get('user')
            
            print_success("Login successful")
            print(f"  Email: {user['email']}")
            print(f"  Name: {user['full_name']}")
            print(f"  Role: {user['role']}")
            print(f"  Token: {token[:20]}...")
            
            return token
        else:
            print_error(f"Login failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return None

# ============================================================================
# TEST 4: Get Current User
# ============================================================================
def test_get_current_user(token):
    print_header("TEST 4: Get Current User")
    
    if not token:
        print_error("No token available")
        return False
    
    try:
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Current user retrieved successfully")
            print(f"  Email: {data['email']}")
            print(f"  Name: {data['full_name']}")
            print(f"  Role: {data['role']}")
            return True
        else:
            print_error(f"Failed to get current user: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ============================================================================
# TEST 5: Refresh Token
# ============================================================================
def test_refresh_token(token):
    print_header("TEST 5: Refresh Token")
    
    if not token:
        print_error("No token available")
        return None
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/refresh-token",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            new_token = data.get('access_token')
            
            print_success("Token refreshed successfully")
            print(f"  New Token: {new_token[:20]}...")
            print(f"  Token Type: {data.get('token_type')}")
            
            return new_token
        else:
            print_error(f"Token refresh failed: {response.status_code}")
            return None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return None

# ============================================================================
# TEST 6: Protected Endpoint
# ============================================================================
def test_protected_endpoint(token):
    print_header("TEST 6: Protected Endpoint")
    
    if not token:
        print_error("No token available")
        return False
    
    try:
        response = requests.get(
            f"{BASE_URL}/auth/protected-example",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Protected endpoint accessed successfully")
            print(f"  Message: {data['message']}")
            print(f"  Role: {data['role']}")
            return True
        else:
            print_error(f"Failed to access protected endpoint: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ============================================================================
# TEST 7: Invalid Token
# ============================================================================
def test_invalid_token():
    print_header("TEST 7: Invalid Token Handling")
    
    invalid_token = "invalid.token.here"
    
    try:
        response = requests.get(
            f"{BASE_URL}/auth/me",
            headers={"Authorization": f"Bearer {invalid_token}"}
        )
        
        if response.status_code == 401:
            print_success("Invalid token correctly rejected")
            print(f"  Status: {response.status_code}")
            print(f"  Message: {response.json().get('detail', 'Unauthorized')}")
            return True
        else:
            print_error(f"Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ============================================================================
# TEST 8: Invalid Login Credentials
# ============================================================================
def test_invalid_credentials():
    print_header("TEST 8: Invalid Login Credentials")
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "wrongpassword"
            }
        )
        
        if response.status_code == 401:
            print_success("Invalid credentials correctly rejected")
            print(f"  Status: {response.status_code}")
            print(f"  Message: {response.json().get('detail', 'Unauthorized')}")
            return True
        else:
            print_error(f"Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ============================================================================
# TEST 9: Logout
# ============================================================================
def test_logout(token):
    print_header("TEST 9: Logout")
    
    if not token:
        print_error("No token available")
        return False
    
    try:
        response = requests.post(
            f"{BASE_URL}/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success("Logout successful")
            print(f"  Message: {data.get('message', 'Logged out')}")
            return True
        else:
            print_error(f"Logout failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Smart DentalOps - Authentication System Test Suite{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    results = {
        "demo_credentials": False,
        "registration": False,
        "login": False,
        "get_current_user": False,
        "refresh_token": False,
        "protected_endpoint": False,
        "invalid_token": False,
        "invalid_credentials": False,
        "logout": False
    }
    
    # Test 1: Get demo credentials
    demo_users = test_demo_credentials()
    results["demo_credentials"] = len(demo_users) > 0
    
    # Test 2: Registration
    new_user = test_registration()
    results["registration"] = new_user is not None
    
    # Test 3: Login
    token = test_login(demo_users)
    results["login"] = token is not None
    
    # Test 4: Get current user
    if token:
        results["get_current_user"] = test_get_current_user(token)
    
    # Test 5: Refresh token
    if token:
        new_token = test_refresh_token(token)
        results["refresh_token"] = new_token is not None
        if new_token:
            token = new_token
    
    # Test 6: Protected endpoint
    if token:
        results["protected_endpoint"] = test_protected_endpoint(token)
    
    # Test 7: Invalid token
    results["invalid_token"] = test_invalid_token()
    
    # Test 8: Invalid credentials
    results["invalid_credentials"] = test_invalid_credentials()
    
    # Test 9: Logout
    if token:
        results["logout"] = test_logout(token)
    
    # Print summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    if passed == total:
        print_success("All authentication tests passed!")
    else:
        print_error(f"{total - passed} test(s) failed")

if __name__ == "__main__":
    main()
