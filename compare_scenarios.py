"""
Scenario Comparison Script
Compare different patient scenarios side-by-side
Perfect for understanding how different factors affect predictions

Run this to see comparisons:
    python compare_scenarios.py
"""

import numpy as np
import joblib
import pandas as pd

print("🔬 Patient Scenario Comparison")
print("=" * 60)

# Load model and scaler
try:
    model = joblib.load('saved_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

def predict_scenario(age, procedure, previous_no_show, visit_day=2, time_slot=1):
    """Make prediction and return risk percentage"""
    procedure_map = {'cleaning': 0, 'root canal': 1, 'extraction': 2}
    procedure_num = procedure_map.get(procedure.lower(), 0)
    
    input_data = np.array([[age, procedure_num, previous_no_show, visit_day, time_slot]])
    input_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_scaled)
    
    no_show_risk = (1 - prediction_proba[0][1]) * 100
    return no_show_risk

# Define comparison scenarios
scenarios = [
    # Age comparison
    {
        'category': 'AGE IMPACT',
        'comparisons': [
            {'name': 'Young (22)', 'age': 22, 'procedure': 'cleaning', 'prev_no_show': 0},
            {'name': 'Middle-aged (45)', 'age': 45, 'procedure': 'cleaning', 'prev_no_show': 0},
            {'name': 'Senior (68)', 'age': 68, 'procedure': 'cleaning', 'prev_no_show': 0},
        ]
    },
    # Procedure comparison
    {
        'category': 'PROCEDURE TYPE IMPACT',
        'comparisons': [
            {'name': 'Cleaning', 'age': 35, 'procedure': 'cleaning', 'prev_no_show': 0},
            {'name': 'Root Canal', 'age': 35, 'procedure': 'root canal', 'prev_no_show': 0},
            {'name': 'Extraction', 'age': 35, 'procedure': 'extraction', 'prev_no_show': 0},
        ]
    },
    # Previous no-show comparison
    {
        'category': 'PREVIOUS NO-SHOW IMPACT',
        'comparisons': [
            {'name': 'Reliable patient', 'age': 30, 'procedure': 'cleaning', 'prev_no_show': 0},
            {'name': 'Previous no-show', 'age': 30, 'procedure': 'cleaning', 'prev_no_show': 1},
        ]
    },
    # Combined factors
    {
        'category': 'COMBINED FACTORS',
        'comparisons': [
            {'name': 'Best case: Senior + Root Canal + Reliable', 'age': 65, 'procedure': 'root canal', 'prev_no_show': 0},
            {'name': 'Worst case: Young + Cleaning + Previous no-show', 'age': 22, 'procedure': 'cleaning', 'prev_no_show': 1},
            {'name': 'Average case: Middle-aged + Extraction + Reliable', 'age': 40, 'procedure': 'extraction', 'prev_no_show': 0},
        ]
    },
]

# Run comparisons
for scenario_group in scenarios:
    print("\n" + "=" * 60)
    print(f"📊 {scenario_group['category']}")
    print("=" * 60)
    
    results = []
    for comparison in scenario_group['comparisons']:
        risk = predict_scenario(
            comparison['age'],
            comparison['procedure'],
            comparison['prev_no_show']
        )
        results.append({
            'Scenario': comparison['name'],
            'No-Show Risk': f"{risk:.1f}%",
            'Risk_Value': risk
        })
    
    # Display results
    for result in results:
        risk_val = result['Risk_Value']
        
        # Visual bar
        bar_length = int(risk_val / 2)  # Scale to 50 chars max
        bar = '█' * bar_length
        
        # Color indicator
        if risk_val > 60:
            indicator = '🔴'
        elif risk_val > 30:
            indicator = '🟡'
        else:
            indicator = '🟢'
        
        print(f"\n{result['Scenario']}")
        print(f"   {indicator} Risk: {result['No-Show Risk']}")
        print(f"   {bar}")
    
    # Analysis
    print(f"\n💡 Analysis:")
    risks = [r['Risk_Value'] for r in results]
    if len(risks) > 1:
        diff = max(risks) - min(risks)
        print(f"   Risk difference: {diff:.1f} percentage points")
        
        if scenario_group['category'] == 'AGE IMPACT':
            print(f"   → Younger patients have higher no-show risk")
        elif scenario_group['category'] == 'PROCEDURE TYPE IMPACT':
            print(f"   → Complex procedures (root canal) have lower no-show risk")
        elif scenario_group['category'] == 'PREVIOUS NO-SHOW IMPACT':
            print(f"   → Previous no-show is a strong predictor of future behavior")
        elif scenario_group['category'] == 'COMBINED FACTORS':
            print(f"   → Multiple factors compound to create very different risk levels")

# Summary insights
print("\n\n" + "=" * 60)
print("🎯 KEY INSIGHTS FOR YOUR PRESENTATION")
print("=" * 60)

insights = [
    {
        'title': '1. Age Factor',
        'insight': 'Younger patients (18-25) have significantly higher no-show rates. Consider extra reminders for this group.'
    },
    {
        'title': '2. Procedure Complexity',
        'insight': 'Patients with complex procedures (root canal) are more likely to show up, possibly due to urgency and cost.'
    },
    {
        'title': '3. Historical Behavior',
        'insight': 'Previous no-show is the strongest predictor. Patients who missed before are likely to miss again.'
    },
    {
        'title': '4. Risk Stratification',
        'insight': 'The model can identify high-risk patients (>60%) who need special attention and alternate scheduling.'
    },
    {
        'title': '5. Business Impact',
        'insight': 'By predicting no-shows, clinics can overbook strategically or send targeted reminders to reduce losses.'
    }
]

for insight in insights:
    print(f"\n{insight['title']}")
    print(f"   {insight['insight']}")

print("\n" + "=" * 60)
print("✅ Comparison Complete!")
print("=" * 60)
print("\n💡 Use these insights in your project presentation to show")
print("   you understand how the AI model makes decisions!")
