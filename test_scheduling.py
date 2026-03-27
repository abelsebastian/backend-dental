"""
Test Script for Scheduling Recommendation Engine
Tests the new scheduling module with different risk scenarios

Run this to verify the scheduling engine works correctly:
    python test_scheduling.py
"""

import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

print("🧪 Testing Scheduling Recommendation Engine")
print("=" * 60)

# Test scenarios with expected slot types
test_scenarios = [
    {
        "name": "Low Risk Patient - Should get Standard Slot",
        "patient": {
            "name": "Alice Johnson",
            "age": 55,
            "procedure": "root canal",
            "previousNoShow": False
        },
        "expected_slot_type": "Standard Slot"
    },
    {
        "name": "Medium Risk Patient - Should get Confirmation Slot",
        "patient": {
            "name": "Bob Smith",
            "age": 28,
            "procedure": "cleaning",
            "previousNoShow": False
        },
        "expected_slot_type": "Confirmation Slot"
    },
    {
        "name": "High Risk Patient - Should get Backup Slot",
        "patient": {
            "name": "Charlie Brown",
            "age": 22,
            "procedure": "cleaning",
            "previousNoShow": True
        },
        "expected_slot_type": "Backup Slot"
    },
    {
        "name": "Very Low Risk - Senior with Root Canal",
        "patient": {
            "name": "Diana Prince",
            "age": 68,
            "procedure": "root canal",
            "previousNoShow": False
        },
        "expected_slot_type": "Standard Slot"
    },
    {
        "name": "Borderline Medium Risk",
        "patient": {
            "name": "Edward Norton",
            "age": 40,
            "procedure": "cleaning",
            "previousNoShow": False
        },
        "expected_slot_type": "Confirmation Slot"
    }
]

# Run tests
passed = 0
failed = 0

for i, scenario in enumerate(test_scenarios, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}: {scenario['name']}")
    print(f"{'='*60}")
    
    try:
        # Make prediction request
        response = requests.post(f"{BASE_URL}/predict", json=scenario['patient'])
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            print(f"\n📋 Patient: {scenario['patient']['name']}")
            print(f"   Age: {scenario['patient']['age']}")
            print(f"   Procedure: {scenario['patient']['procedure']}")
            print(f"   Previous No-Show: {scenario['patient']['previousNoShow']}")
            
            print(f"\n🤖 AI Prediction:")
            print(f"   No-Show Risk: {result['risk']}")
            print(f"   Duration: {result['duration']}")
            
            print(f"\n🎯 Scheduling Recommendation:")
            print(f"   Slot Time: {result['slot']}")
            print(f"   Slot Type: {result['slotType']}")
            print(f"   Reason: {result['slotReason']}")
            
            # Verify expected slot type
            if result['slotType'] == scenario['expected_slot_type']:
                print(f"\n✅ PASS - Got expected slot type: {result['slotType']}")
                passed += 1
            else:
                print(f"\n❌ FAIL - Expected {scenario['expected_slot_type']}, got {result['slotType']}")
                failed += 1
                
        else:
            print(f"❌ FAIL - HTTP {response.status_code}")
            print(f"Response: {response.text}")
            failed += 1
            
    except Exception as e:
        print(f"❌ FAIL - Error: {e}")
        failed += 1

# Summary
print("\n" + "=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)
print(f"Total Tests: {len(test_scenarios)}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"Success Rate: {(passed/len(test_scenarios)*100):.1f}%")

if failed == 0:
    print("\n🎉 All tests passed! Scheduling engine working correctly.")
else:
    print(f"\n⚠️ {failed} test(s) failed. Check the output above for details.")

print("\n" + "=" * 60)
print("💡 Tips:")
print("   • Make sure backend is running: uvicorn main:app --reload")
print("   • Check http://localhost:8000/docs for API documentation")
print("   • Slot types depend on ANN risk prediction")
print("=" * 60)
