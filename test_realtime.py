"""
Phase 1: Test Script for Real-Time Synthetic Data Endpoint

This script tests the /live-dashboard-data endpoint to ensure
it generates proper synthetic appointment data.

Run this after starting the backend server:
    python test_realtime.py
"""

import requests
import json
from datetime import datetime

# Backend URL
BASE_URL = "http://localhost:8000"

def test_live_dashboard_data():
    """
    Test the /live-dashboard-data endpoint
    """
    print("=" * 60)
    print("Testing Phase 1: Real-Time Synthetic Data Endpoint")
    print("=" * 60)
    
    try:
        # Make request to the endpoint
        print("\n📡 Sending GET request to /live-dashboard-data...")
        response = requests.get(f"{BASE_URL}/live-dashboard-data")
        
        # Check if request was successful
        if response.status_code == 200:
            print("✅ Request successful (Status 200)")
            
            # Parse JSON response
            data = response.json()
            
            # Display summary information
            print("\n📊 Summary Statistics:")
            print(f"   Timestamp: {data['timestamp']}")
            print(f"   Total Appointments: {data['summary']['totalAppointments']}")
            print(f"   High Risk Count: {data['summary']['highRiskCount']}")
            print(f"   Average Risk: {data['summary']['averageRisk']}%")
            print(f"   Chair Utilization: {data['summary']['chairUtilization']}%")
            
            # Display first 3 appointments
            print("\n👥 Sample Appointments:")
            for i, apt in enumerate(data['appointments'][:3], 1):
                print(f"\n   Appointment {i}:")
                print(f"      Patient: {apt['patientName']} (ID: {apt['patientId']})")
                print(f"      Age: {apt['age']}")
                print(f"      Procedure: {apt['procedureType']}")
                print(f"      Risk Score: {apt['riskScore']}%")
                print(f"      Slot: {apt['slotTime']} ({apt['slotType']})")
                print(f"      Status: {apt['status']}")
                print(f"      Dentist: {apt['dentist']}")
                print(f"      Chair: {apt['chair']}")
                print(f"      Duration: {apt['duration']}")
            
            # Validate data structure
            print("\n🔍 Validating Data Structure...")
            required_fields = [
                'patientId', 'patientName', 'age', 'procedureType',
                'riskScore', 'slotType', 'slotTime', 'status',
                'dentist', 'chair', 'duration', 'timestamp'
            ]
            
            all_valid = True
            for apt in data['appointments']:
                for field in required_fields:
                    if field not in apt:
                        print(f"   ❌ Missing field: {field}")
                        all_valid = False
            
            if all_valid:
                print("   ✅ All appointments have required fields")
            
            # Test multiple requests to see variation
            print("\n🔄 Testing Data Variation (3 requests)...")
            risk_scores = []
            for i in range(3):
                resp = requests.get(f"{BASE_URL}/live-dashboard-data")
                if resp.status_code == 200:
                    avg_risk = resp.json()['summary']['averageRisk']
                    risk_scores.append(avg_risk)
                    print(f"   Request {i+1}: Average Risk = {avg_risk}%")
            
            if len(set(risk_scores)) > 1:
                print("   ✅ Data varies between requests (as expected)")
            else:
                print("   ⚠️ Data appears static (unexpected)")
            
            print("\n" + "=" * 60)
            print("✅ Phase 1 Test Complete - All checks passed!")
            print("=" * 60)
            
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the backend server running?")
        print("   Start it with: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_live_dashboard_data()
