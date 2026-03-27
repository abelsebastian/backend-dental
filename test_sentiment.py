"""
Phase 2: Test Script for Sentiment Analysis Endpoint

This script tests the /analyze-sentiment endpoint with various
patient messages to verify sentiment detection and risk adjustment.

Run this after starting the backend server:
    python test_sentiment.py
"""

import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

# Test cases with different sentiments
TEST_CASES = [
    {
        "name": "Positive Message",
        "message": "I'm really looking forward to my appointment! Thank you for the reminder.",
        "currentRisk": 50.0,
        "expected_sentiment": "positive"
    },
    {
        "name": "Negative Message",
        "message": "I'm very frustrated with the long wait times. This is unacceptable.",
        "currentRisk": 50.0,
        "expected_sentiment": "negative"
    },
    {
        "name": "Neutral Message",
        "message": "I received your message about the appointment on Tuesday.",
        "currentRisk": 50.0,
        "expected_sentiment": "neutral"
    },
    {
        "name": "Cancellation Intent (Negative)",
        "message": "I don't think I can make it. I'm not happy with the service.",
        "currentRisk": 60.0,
        "expected_sentiment": "negative"
    },
    {
        "name": "Confirmation (Positive)",
        "message": "Yes, I'll be there! Excited to see Dr. Smith.",
        "currentRisk": 30.0,
        "expected_sentiment": "positive"
    },
    {
        "name": "Delay Request (Neutral)",
        "message": "Can we reschedule to next week?",
        "currentRisk": 40.0,
        "expected_sentiment": "neutral"
    },
    {
        "name": "High Risk with Negative Sentiment",
        "message": "I'm really annoyed. This is terrible service.",
        "currentRisk": 75.0,
        "expected_sentiment": "negative"
    }
]

def test_sentiment_analysis():
    """
    Test the /analyze-sentiment endpoint with various messages
    """
    print("=" * 70)
    print("Testing Phase 2: Sentiment Analysis Endpoint")
    print("=" * 70)
    
    try:
        # Test each case
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{'='*70}")
            print(f"Test Case {i}: {test_case['name']}")
            print(f"{'='*70}")
            print(f"Message: \"{test_case['message']}\"")
            print(f"Current Risk: {test_case['currentRisk']}%")
            print(f"Expected Sentiment: {test_case['expected_sentiment']}")
            
            # Make request
            payload = {
                "message": test_case["message"],
                "currentRisk": test_case["currentRisk"]
            }
            
            response = requests.post(
                f"{BASE_URL}/analyze-sentiment",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n✅ Response:")
                print(f"   Sentiment: {data['sentiment']}")
                print(f"   Polarity: {data['polarity']}")
                print(f"   Subjectivity: {data['subjectivity']}")
                print(f"   Adjusted Risk: {data['adjustedRisk']}%")
                print(f"   Risk Change: {data['riskChange']:+.1f}%")
                print(f"   Explanation: {data['explanation']}")
                
                # Verify sentiment matches expected
                if data['sentiment'] == test_case['expected_sentiment']:
                    print(f"\n   ✅ Sentiment matches expected: {test_case['expected_sentiment']}")
                else:
                    print(f"\n   ⚠️ Sentiment mismatch: Expected {test_case['expected_sentiment']}, got {data['sentiment']}")
                
                # Verify risk adjustment logic
                if data['sentiment'] == 'negative':
                    expected_change = 15.0
                    if abs(data['riskChange'] - expected_change) < 0.1:
                        print(f"   ✅ Risk adjustment correct: +{expected_change}%")
                    else:
                        print(f"   ❌ Risk adjustment incorrect: Expected +{expected_change}%, got {data['riskChange']:+.1f}%")
                else:
                    if data['riskChange'] == 0.0:
                        print(f"   ✅ Risk unchanged (as expected for {data['sentiment']} sentiment)")
                    else:
                        print(f"   ❌ Risk should be unchanged for {data['sentiment']} sentiment")
                
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ Phase 2 Sentiment Analysis Test Complete!")
        print(f"{'='*70}")
        print(f"\nTested {len(TEST_CASES)} cases covering:")
        print("  • Positive sentiment (no risk change)")
        print("  • Neutral sentiment (no risk change)")
        print("  • Negative sentiment (+15% risk)")
        print("  • Various risk levels (30% to 75%)")
        print("  • Different message types (confirmation, cancellation, delay)")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the backend server running?")
        print("   Start it with: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error during test: {e}")

def test_edge_cases():
    """
    Test edge cases and error handling
    """
    print(f"\n{'='*70}")
    print("Testing Edge Cases")
    print(f"{'='*70}")
    
    edge_cases = [
        {
            "name": "Empty Message",
            "payload": {"message": "", "currentRisk": 50.0},
            "should_fail": True
        },
        {
            "name": "Very Long Message",
            "payload": {
                "message": "This is a very long message. " * 50,
                "currentRisk": 50.0
            },
            "should_fail": False
        },
        {
            "name": "Invalid Risk (Negative)",
            "payload": {"message": "Test message", "currentRisk": -10.0},
            "should_fail": True
        },
        {
            "name": "Invalid Risk (Over 100)",
            "payload": {"message": "Test message", "currentRisk": 150.0},
            "should_fail": True
        },
        {
            "name": "Risk at 100% with Negative Sentiment",
            "payload": {"message": "I'm very unhappy", "currentRisk": 100.0},
            "should_fail": False
        }
    ]
    
    for test in edge_cases:
        print(f"\n{test['name']}:")
        response = requests.post(f"{BASE_URL}/analyze-sentiment", json=test["payload"])
        
        if test["should_fail"]:
            if response.status_code != 200:
                print(f"   ✅ Correctly rejected (Status {response.status_code})")
            else:
                print(f"   ❌ Should have failed but succeeded")
        else:
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Handled correctly")
                if test["name"] == "Risk at 100% with Negative Sentiment":
                    print(f"      Risk capped at: {data['adjustedRisk']}%")
            else:
                print(f"   ❌ Should have succeeded but failed (Status {response.status_code})")

if __name__ == "__main__":
    test_sentiment_analysis()
    test_edge_cases()
