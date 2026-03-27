"""
Phase 3: Test Script for Intent Detection Endpoint

This script tests the /detect-intent endpoint with various
patient messages to verify intent detection and risk adjustment.

Run this after starting the backend server:
    python test_intent.py
"""

import requests
import json

# Backend URL
BASE_URL = "http://localhost:8000"

# Test cases with different intents
TEST_CASES = [
    {
        "name": "Clear Confirmation",
        "message": "Yes, I confirm my appointment. I'll be there!",
        "currentRisk": 50.0,
        "expected_intent": "Confirmation"
    },
    {
        "name": "Clear Cancellation",
        "message": "I need to cancel my appointment. I can't make it.",
        "currentRisk": 50.0,
        "expected_intent": "Cancellation"
    },
    {
        "name": "Clear Delay/Reschedule",
        "message": "I'm running late. Can we reschedule?",
        "currentRisk": 50.0,
        "expected_intent": "Delay"
    },
    {
        "name": "Excited Confirmation",
        "message": "Looking forward to my appointment! See you tomorrow!",
        "currentRisk": 40.0,
        "expected_intent": "Confirmation"
    },
    {
        "name": "Cancellation with Reason",
        "message": "I won't be able to make it. Something came up.",
        "currentRisk": 60.0,
        "expected_intent": "Cancellation"
    },
    {
        "name": "Reschedule Request",
        "message": "Can we move the appointment to next week?",
        "currentRisk": 45.0,
        "expected_intent": "Delay"
    },
    {
        "name": "Uncertain (Weak Delay)",
        "message": "I'm not sure if I can make it.",
        "currentRisk": 50.0,
        "expected_intent": "Delay"
    },
    {
        "name": "Weak Confirmation",
        "message": "Okay, I guess I'll be there.",
        "currentRisk": 50.0,
        "expected_intent": "Confirmation"
    },
    {
        "name": "Unknown Intent",
        "message": "I received your message.",
        "currentRisk": 50.0,
        "expected_intent": "Unknown"
    },
    {
        "name": "High Risk + Cancellation",
        "message": "I have to cancel. Sorry.",
        "currentRisk": 80.0,
        "expected_intent": "Cancellation"
    }
]

def test_intent_detection():
    """
    Test the /detect-intent endpoint with various messages
    """
    print("=" * 70)
    print("Testing Phase 3: Intent Detection Endpoint")
    print("=" * 70)
    
    try:
        # Test each case
        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{'='*70}")
            print(f"Test Case {i}: {test_case['name']}")
            print(f"{'='*70}")
            print(f"Message: \"{test_case['message']}\"")
            print(f"Current Risk: {test_case['currentRisk']}%")
            print(f"Expected Intent: {test_case['expected_intent']}")
            
            # Make request
            payload = {
                "message": test_case["message"],
                "currentRisk": test_case["currentRisk"]
            }
            
            response = requests.post(
                f"{BASE_URL}/detect-intent",
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n✅ Response:")
                print(f"   Intent: {data['intent']}")
                print(f"   Confidence: {data['confidence']}")
                print(f"   Keywords: {', '.join(data['keywords']) if data['keywords'] else 'None'}")
                print(f"   Adjusted Risk: {data['adjustedRisk']}%")
                print(f"   Risk Change: {data['riskChange']:+.1f}%")
                print(f"   Explanation: {data['explanation']}")
                
                # Verify intent matches expected
                if data['intent'] == test_case['expected_intent']:
                    print(f"\n   ✅ Intent matches expected: {test_case['expected_intent']}")
                else:
                    print(f"\n   ⚠️ Intent mismatch: Expected {test_case['expected_intent']}, got {data['intent']}")
                
                # Verify risk adjustment logic
                if data['intent'] == 'Confirmation':
                    if data['riskChange'] <= 0:
                        print(f"   ✅ Risk decreased (as expected for confirmation)")
                    else:
                        print(f"   ❌ Risk should decrease for confirmation")
                elif data['intent'] == 'Cancellation':
                    if data['riskChange'] > 0:
                        print(f"   ✅ Risk increased (as expected for cancellation)")
                    else:
                        print(f"   ❌ Risk should increase for cancellation")
                elif data['intent'] == 'Delay':
                    if data['riskChange'] > 0:
                        print(f"   ✅ Risk increased (as expected for delay)")
                    else:
                        print(f"   ❌ Risk should increase for delay")
                else:  # Unknown
                    if data['riskChange'] == 0:
                        print(f"   ✅ Risk unchanged (as expected for unknown intent)")
                    else:
                        print(f"   ❌ Risk should be unchanged for unknown intent")
                
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        
        # Summary
        print(f"\n{'='*70}")
        print("✅ Phase 3 Intent Detection Test Complete!")
        print(f"{'='*70}")
        print(f"\nTested {len(TEST_CASES)} cases covering:")
        print("  • Confirmation intent (risk -10%)")
        print("  • Cancellation intent (risk +25%)")
        print("  • Delay/reschedule intent (risk +5%)")
        print("  • Unknown intent (no change)")
        print("  • Various confidence levels (High, Medium, Low)")
        print("  • Different risk levels (40% to 80%)")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Is the backend server running?")
        print("   Start it with: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ Error during test: {e}")

def test_combined_sentiment_and_intent():
    """
    Test combining sentiment analysis with intent detection
    """
    print(f"\n{'='*70}")
    print("Testing Combined Sentiment + Intent Analysis")
    print(f"{'='*70}")
    
    combined_cases = [
        {
            "name": "Negative Sentiment + Cancellation",
            "message": "I'm frustrated and need to cancel.",
            "initial_risk": 50.0
        },
        {
            "name": "Positive Sentiment + Confirmation",
            "message": "I'm excited! Yes, I'll be there!",
            "initial_risk": 50.0
        },
        {
            "name": "Neutral Sentiment + Delay",
            "message": "Can we reschedule to next week?",
            "initial_risk": 50.0
        }
    ]
    
    for test in combined_cases:
        print(f"\n{test['name']}:")
        print(f"Message: \"{test['message']}\"")
        print(f"Initial Risk: {test['initial_risk']}%")
        
        # Step 1: Sentiment analysis
        sentiment_response = requests.post(
            f"{BASE_URL}/analyze-sentiment",
            json={"message": test["message"], "currentRisk": test["initial_risk"]}
        )
        
        if sentiment_response.status_code == 200:
            sentiment_data = sentiment_response.json()
            print(f"\n  Step 1 - Sentiment Analysis:")
            print(f"    Sentiment: {sentiment_data['sentiment']}")
            print(f"    Risk after sentiment: {sentiment_data['adjustedRisk']}%")
            
            # Step 2: Intent detection (using risk after sentiment)
            intent_response = requests.post(
                f"{BASE_URL}/detect-intent",
                json={"message": test["message"], "currentRisk": sentiment_data['adjustedRisk']}
            )
            
            if intent_response.status_code == 200:
                intent_data = intent_response.json()
                print(f"\n  Step 2 - Intent Detection:")
                print(f"    Intent: {intent_data['intent']}")
                print(f"    Final Risk: {intent_data['adjustedRisk']}%")
                
                print(f"\n  📊 Summary:")
                print(f"    Initial Risk: {test['initial_risk']}%")
                print(f"    After Sentiment: {sentiment_data['adjustedRisk']}% ({sentiment_data['riskChange']:+.1f}%)")
                print(f"    After Intent: {intent_data['adjustedRisk']}% ({intent_data['riskChange']:+.1f}%)")
                print(f"    Total Change: {intent_data['adjustedRisk'] - test['initial_risk']:+.1f}%")

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
            "name": "Multiple Intents (Cancellation Priority)",
            "payload": {"message": "I need to cancel but maybe reschedule", "currentRisk": 50.0},
            "should_fail": False
        },
        {
            "name": "Risk at 0% with Confirmation",
            "payload": {"message": "Yes, I confirm!", "currentRisk": 0.0},
            "should_fail": False
        },
        {
            "name": "Risk at 100% with Cancellation",
            "payload": {"message": "I have to cancel", "currentRisk": 100.0},
            "should_fail": False
        }
    ]
    
    for test in edge_cases:
        print(f"\n{test['name']}:")
        response = requests.post(f"{BASE_URL}/detect-intent", json=test["payload"])
        
        if test["should_fail"]:
            if response.status_code != 200:
                print(f"   ✅ Correctly rejected (Status {response.status_code})")
            else:
                print(f"   ❌ Should have failed but succeeded")
        else:
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Handled correctly")
                print(f"      Intent: {data['intent']}")
                print(f"      Risk: {data['adjustedRisk']}%")
            else:
                print(f"   ❌ Should have succeeded but failed (Status {response.status_code})")

if __name__ == "__main__":
    test_intent_detection()
    test_combined_sentiment_and_intent()
    test_edge_cases()
