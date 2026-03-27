"""
Interactive Demo Script
Test the neural network with your own patient data interactively

Run this to test different scenarios:
    python interactive_demo.py
"""

import numpy as np
import joblib

print("🎮 Interactive Neural Network Demo")
print("=" * 60)
print("Test the AI model with different patient scenarios!\n")

# Load model and scaler
try:
    model = joblib.load('saved_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please run model.py first to train the model.")
    exit()

def get_risk_level(risk_percentage):
    """Determine risk level and color"""
    if risk_percentage > 60:
        return "HIGH RISK ⚠️", "Consider alternate slot or send multiple reminders"
    elif risk_percentage > 30:
        return "MEDIUM RISK ⚡", "Send reminder notification 24 hours before"
    else:
        return "LOW RISK ✅", "Standard confirmation is sufficient"

def predict_patient(age, procedure, previous_no_show, visit_day=2, time_slot=1):
    """Make prediction for a patient"""
    
    # Map procedure to number
    procedure_map = {
        'cleaning': 0,
        'root canal': 1,
        'extraction': 2
    }
    procedure_num = procedure_map.get(procedure.lower(), 0)
    
    # Create input array
    input_data = np.array([[age, procedure_num, previous_no_show, visit_day, time_slot]])
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_scaled)
    
    show_up_prob = prediction_proba[0][1] * 100
    no_show_risk = (1 - prediction_proba[0][1]) * 100
    
    return show_up_prob, no_show_risk

# Interactive mode
print("=" * 60)
print("INTERACTIVE MODE - Enter patient details")
print("=" * 60)

while True:
    print("\n" + "-" * 60)
    
    # Get patient details
    try:
        print("\n📋 Enter Patient Details:")
        
        age = input("Age (18-80): ").strip()
        if age.lower() == 'quit' or age.lower() == 'exit':
            print("\n👋 Thanks for using the demo!")
            break
        age = int(age)
        
        if age < 18 or age > 80:
            print("⚠️ Age should be between 18 and 80")
            continue
        
        print("\nProcedure Type:")
        print("  1. Cleaning")
        print("  2. Root Canal")
        print("  3. Extraction")
        procedure_choice = input("Choose (1-3): ").strip()
        
        procedure_map_input = {
            '1': 'cleaning',
            '2': 'root canal',
            '3': 'extraction'
        }
        procedure = procedure_map_input.get(procedure_choice, 'cleaning')
        
        previous_no_show_input = input("\nPrevious No-Show? (yes/no): ").strip().lower()
        previous_no_show = 1 if previous_no_show_input in ['yes', 'y', '1'] else 0
        
        # Make prediction
        show_up_prob, no_show_risk = predict_patient(age, procedure, previous_no_show)
        
        # Display results
        print("\n" + "=" * 60)
        print("🤖 AI PREDICTION RESULTS")
        print("=" * 60)
        
        print(f"\n👤 Patient Profile:")
        print(f"   Age: {age} years")
        print(f"   Procedure: {procedure.title()}")
        print(f"   Previous No-Show: {'Yes' if previous_no_show else 'No'}")
        
        print(f"\n📊 Prediction:")
        print(f"   Show-Up Probability: {show_up_prob:.1f}%")
        print(f"   No-Show Risk: {no_show_risk:.1f}%")
        
        risk_level, recommendation = get_risk_level(no_show_risk)
        print(f"\n🎯 Risk Level: {risk_level}")
        print(f"💡 Recommendation: {recommendation}")
        
        # Duration based on procedure
        duration_map = {
            'cleaning': '20 minutes',
            'root canal': '45 minutes',
            'extraction': '30 minutes'
        }
        duration = duration_map.get(procedure, '30 minutes')
        print(f"⏱️ Estimated Duration: {duration}")
        
        # Suggested slot
        if no_show_risk > 50:
            slot = "2:00 PM (Alternate slot - High risk patient)"
        else:
            slot = "10:30 AM (Preferred slot)"
        print(f"📅 Suggested Slot: {slot}")
        
        print("\n" + "=" * 60)
        
        # Continue or exit
        continue_choice = input("\nTest another patient? (yes/no): ").strip().lower()
        if continue_choice not in ['yes', 'y', '1']:
            print("\n👋 Thanks for using the demo!")
            break
            
    except ValueError:
        print("⚠️ Invalid input. Please enter valid numbers.")
        continue
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the demo!")
        break
    except Exception as e:
        print(f"⚠️ Error: {e}")
        continue

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
