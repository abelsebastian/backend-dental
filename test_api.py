"""
Simple API Testing Script
Run this to test if the backend is working correctly

Usage: python test_api.py
"""

import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

print("🧪 Testing Smart DentalOps Backend API")
print("=" * 60)

# Test 1: Health Check
print("\n📡 Test 1: Health Check (GET /)")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Health check passed!")
except Exception as e:
    print(f"❌ Health check failed: {e}")

# Test 2: Model Info
print("\n📊 Test 2: Model Info (GET /model-info)")
try:
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Model info retrieved!")
except Exception as e:
    print(f"❌ Model info failed: {e}")

# Test 3: Prediction - Low Risk Patient
print("\n🔮 Test 3: Prediction - Low Risk Patient")
try:
    patient_data = {
        "name": "Alice Johnson",
        "age": 45,
        "procedure": "root canal",
        "previousNoShow": False
    }
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"Status Code: {response.status_code}")
    print(f"Patient: {patient_data['name']}, Age: {patient_data['age']}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Prediction successful!")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test 4: Prediction - High Risk Patient
print("\n🔮 Test 4: Prediction - High Risk Patient")
try:
    patient_data = {
        "name": "Bob Smith",
        "age": 22,
        "procedure": "cleaning",
        "previousNoShow": True
    }
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"Status Code: {response.status_code}")
    print(f"Patient: {patient_data['name']}, Age: {patient_data['age']}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Prediction successful!")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test 5: Prediction - Medium Risk Patient
print("\n🔮 Test 5: Prediction - Medium Risk Patient")
try:
    patient_data = {
        "name": "Carol Davis",
        "age": 35,
        "procedure": "extraction",
        "previousNoShow": False
    }
    response = requests.post(f"{BASE_URL}/predict", json=patient_data)
    print(f"Status Code: {response.status_code}")
    print(f"Patient: {patient_data['name']}, Age: {patient_data['age']}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Prediction successful!")
except Exception as e:
    print(f"❌ Prediction failed: {e}")

# Test 6: Invalid Data (should fail validation)
print("\n❌ Test 6: Invalid Data (Testing Error Handling)")
try:
    invalid_data = {
        "name": "Invalid Patient",
        "age": "not a number",  # Should be integer
        "procedure": "cleaning",
        "previousNoShow": False
    }
    response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    if response.status_code == 422:
        print("✅ Validation error handled correctly!")
    else:
        print("⚠️ Expected validation error")
except Exception as e:
    print(f"Expected error: {e}")

print("\n" + "=" * 60)
print("🎉 API Testing Complete!")
print("=" * 60)
print("\n💡 Tips:")
print("   • If tests fail, make sure backend is running: uvicorn main:app --reload")
print("   • If model not found, run: python model.py")
print("   • Check http://localhost:8000/docs for interactive API docs")
