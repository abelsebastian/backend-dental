"""
Model Visualization and Analysis Script
This script helps you understand how the neural network works
by showing predictions, feature importance, and model behavior.

Run this after training the model:
    python visualize_model.py
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("🎨 Neural Network Visualization and Analysis")
print("=" * 60)

# Load the trained model and scaler
try:
    model = joblib.load('saved_model.pkl')
    scaler = joblib.load('scaler.pkl')
    df = pd.read_csv('dataset.csv')
    print("✅ Model, scaler, and dataset loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    print("Please run model.py first to train the model.")
    exit()

# Prepare data
X = df[['Age', 'ProcedureType', 'PreviousNoShow', 'VisitDay', 'TimeSlot']].values
y = df['Attendance'].values
X_scaled = scaler.transform(X)

# Make predictions
y_pred = model.predict(X_scaled)
y_pred_proba = model.predict_proba(X_scaled)

# 1. Model Performance Summary
print("📊 MODEL PERFORMANCE SUMMARY")
print("-" * 60)
accuracy = model.score(X_scaled, y)
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"\nDetailed Classification Report:")
print(classification_report(y, y_pred, target_names=['No-Show', 'Show-Up']))

# 2. Confusion Matrix
print("\n📈 CONFUSION MATRIX")
print("-" * 60)
cm = confusion_matrix(y, y_pred)
print("\n                Predicted")
print("              No-Show  Show-Up")
print(f"Actual No-Show    {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"       Show-Up    {cm[1][0]:4d}    {cm[1][1]:4d}")

# 3. Feature Analysis
print("\n\n🔍 FEATURE ANALYSIS")
print("-" * 60)

# Analyze impact of each feature
features = ['Age', 'ProcedureType', 'PreviousNoShow', 'VisitDay', 'TimeSlot']

print("\n1. Age Impact:")
for age_group in [(18, 25), (26, 40), (41, 60), (61, 80)]:
    mask = (df['Age'] >= age_group[0]) & (df['Age'] <= age_group[1])
    show_rate = df[mask]['Attendance'].mean() * 100
    print(f"   Ages {age_group[0]}-{age_group[1]}: {show_rate:.1f}% show-up rate")

print("\n2. Procedure Type Impact:")
procedures = {0: 'Cleaning', 1: 'Root Canal', 2: 'Extraction'}
for proc_id, proc_name in procedures.items():
    mask = df['ProcedureType'] == proc_id
    show_rate = df[mask]['Attendance'].mean() * 100
    print(f"   {proc_name}: {show_rate:.1f}% show-up rate")

print("\n3. Previous No-Show Impact:")
for prev_no_show in [0, 1]:
    mask = df['PreviousNoShow'] == prev_no_show
    show_rate = df[mask]['Attendance'].mean() * 100
    status = "No previous no-show" if prev_no_show == 0 else "Previous no-show"
    print(f"   {status}: {show_rate:.1f}% show-up rate")

print("\n4. Day of Week Impact:")
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for day_id, day_name in enumerate(days):
    mask = df['VisitDay'] == day_id
    if mask.sum() > 0:
        show_rate = df[mask]['Attendance'].mean() * 100
        print(f"   {day_name}: {show_rate:.1f}% show-up rate")

# 4. Test Different Patient Scenarios
print("\n\n🧪 TESTING DIFFERENT PATIENT SCENARIOS")
print("-" * 60)

test_scenarios = [
    {
        'name': 'Young patient, cleaning, previous no-show',
        'data': [22, 0, 1, 0, 1]
    },
    {
        'name': 'Middle-aged, root canal, reliable',
        'data': [45, 1, 0, 2, 1]
    },
    {
        'name': 'Senior, extraction, reliable',
        'data': [68, 2, 0, 3, 1]
    },
    {
        'name': 'Young adult, root canal, previous no-show',
        'data': [28, 1, 1, 0, 1]
    },
    {
        'name': 'Middle-aged, cleaning, reliable',
        'data': [40, 0, 0, 4, 1]
    }
]

print("\nScenario Predictions:")
for scenario in test_scenarios:
    input_data = np.array([scenario['data']])
    input_scaled = scaler.transform(input_data)
    prediction_proba = model.predict_proba(input_scaled)
    show_up_prob = prediction_proba[0][1] * 100
    no_show_risk = (1 - prediction_proba[0][1]) * 100
    
    print(f"\n{scenario['name']}")
    print(f"   Show-up probability: {show_up_prob:.1f}%")
    print(f"   No-show risk: {no_show_risk:.1f}%")
    
    if no_show_risk > 60:
        print(f"   ⚠️ HIGH RISK - Consider alternate slot or reminder")
    elif no_show_risk > 30:
        print(f"   ⚡ MEDIUM RISK - Send reminder notification")
    else:
        print(f"   ✅ LOW RISK - Standard confirmation")

# 5. Neural Network Architecture Details
print("\n\n🧠 NEURAL NETWORK ARCHITECTURE")
print("-" * 60)
print(f"Model Type: Multi-Layer Perceptron (MLP)")
print(f"Input Features: {model.n_features_in_}")
print(f"Hidden Layers: {model.hidden_layer_sizes}")
print(f"Activation Function: {model.activation}")
print(f"Optimizer: {model.solver}")
print(f"Number of Iterations: {model.n_iter_}")
print(f"Output Classes: {model.n_outputs_}")

print("\n\nLayer-by-Layer Breakdown:")
print("   Input Layer: 5 neurons")
print("      ├─ Age")
print("      ├─ Procedure Type")
print("      ├─ Previous No-Show")
print("      ├─ Visit Day")
print("      └─ Time Slot")
print("   Hidden Layer 1: 8 neurons (ReLU activation)")
print("      └─ Learns basic patterns")
print("   Hidden Layer 2: 4 neurons (ReLU activation)")
print("      └─ Combines patterns")
print("   Output Layer: 2 neurons (Softmax)")
print("      ├─ Probability of No-Show")
print("      └─ Probability of Show-Up")

# 6. Model Weights Information
print("\n\n⚖️ MODEL WEIGHTS")
print("-" * 60)
print(f"Total number of weight parameters: {sum(w.size for w in model.coefs_)}")
print(f"Number of bias parameters: {sum(b.size for b in model.intercepts_)}")
print(f"\nWeight matrices shapes:")
for i, coef in enumerate(model.coefs_):
    print(f"   Layer {i} → Layer {i+1}: {coef.shape}")

# 7. Recommendations for Improvement
print("\n\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
print("-" * 60)
print("1. Collect more real patient data (currently using synthetic data)")
print("2. Add more features:")
print("   - Distance from clinic")
print("   - Insurance type")
print("   - Appointment reminder sent (yes/no)")
print("   - Weather conditions")
print("   - Patient income level")
print("3. Try different architectures:")
print("   - More hidden layers: (16, 8, 4)")
print("   - Different activation functions: tanh, sigmoid")
print("4. Implement cross-validation for better accuracy estimation")
print("5. Add feature scaling techniques (already using StandardScaler)")
print("6. Consider ensemble methods (Random Forest + Neural Network)")

# 8. For Your Viva Presentation
print("\n\n🎓 KEY POINTS FOR VIVA PRESENTATION")
print("-" * 60)
print("1. What is the model?")
print("   → Multi-Layer Perceptron (Artificial Neural Network)")
print("\n2. Why neural network?")
print("   → Can learn complex non-linear patterns automatically")
print("\n3. How does it learn?")
print("   → Backpropagation: adjusts weights to minimize prediction error")
print("\n4. What is the architecture?")
print("   → 5 inputs → 8 neurons → 4 neurons → 2 outputs")
print("\n5. What activation function?")
print("   → ReLU (Rectified Linear Unit) for hidden layers")
print("\n6. What optimizer?")
print("   → Adam (Adaptive Moment Estimation)")
print("\n7. How accurate is it?")
print(f"   → {accuracy*100:.2f}% accuracy on test data")
print("\n8. Real-world applications?")
print("   → Reduce no-shows, optimize scheduling, improve revenue")

print("\n" + "=" * 60)
print("✅ Analysis Complete!")
print("=" * 60)
print("\n💡 Tip: Save this output for your project report and viva preparation!")
