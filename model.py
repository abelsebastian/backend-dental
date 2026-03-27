"""
ANN Model Training Script
This script creates, trains, and saves the Artificial Neural Network
that predicts patient no-show probability.

Uses scikit-learn's MLPClassifier (Multi-Layer Perceptron) which is
a neural network that works with all Python versions.

Run this file ONCE before starting the backend server:
    python model.py
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("🚀 Starting ANN Model Training...")
print("=" * 60)

# Step 1: Generate Synthetic Training Data
print("\n📊 Step 1: Generating training data...")

def generate_training_data(num_samples=1000):
    """
    Generate synthetic dental appointment data for training
    In a real project, this would come from actual clinic records
    """
    np.random.seed(42)  # For reproducibility
    
    data = []
    
    for _ in range(num_samples):
        # Generate random patient data
        age = np.random.randint(18, 81)  # Age between 18-80
        procedure_type = np.random.randint(0, 3)  # 0=cleaning, 1=root canal, 2=extraction
        previous_no_show = np.random.randint(0, 2)  # 0=no, 1=yes
        visit_day = np.random.randint(0, 7)  # 0-6 (Monday to Sunday)
        time_slot = np.random.randint(0, 4)  # 0-3 (different time slots)
        
        # Calculate attendance probability based on factors
        # This simulates real-world patterns
        base_probability = 0.8  # 80% base show-up rate
        
        # Age factor: younger patients less reliable
        if age < 25:
            base_probability -= 0.2
        elif age > 60:
            base_probability += 0.1
        
        # Procedure factor: complex procedures have higher show-up
        if procedure_type == 1:  # root canal
            base_probability += 0.15
        elif procedure_type == 0:  # cleaning
            base_probability -= 0.1
        
        # Previous no-show is strong indicator
        if previous_no_show == 1:
            base_probability -= 0.3
        
        # Day of week factor: Mondays have more no-shows
        if visit_day == 0:  # Monday
            base_probability -= 0.1
        
        # Ensure probability is between 0 and 1
        base_probability = max(0.1, min(0.95, base_probability))
        
        # Determine attendance (1=showed up, 0=no-show)
        attendance = 1 if np.random.random() < base_probability else 0
        
        data.append([age, procedure_type, previous_no_show, visit_day, time_slot, attendance])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'Age', 'ProcedureType', 'PreviousNoShow', 'VisitDay', 'TimeSlot', 'Attendance'
    ])
    
    return df

# Generate data
df = generate_training_data(1000)

# Save to CSV for reference
df.to_csv('dataset.csv', index=False)
print(f"✅ Generated {len(df)} training samples")
print(f"   Show-up rate: {df['Attendance'].mean()*100:.1f}%")
print(f"   No-show rate: {(1-df['Attendance'].mean())*100:.1f}%")

# Step 2: Prepare Data for Training
print("\n🔧 Step 2: Preparing data...")

# Separate features (X) and target (y)
X = df[['Age', 'ProcedureType', 'PreviousNoShow', 'VisitDay', 'TimeSlot']].values
y = df['Attendance'].values

# Normalize features (important for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")

# Step 3: Build Neural Network Architecture
print("\n🧠 Step 3: Building neural network...")

# Create Multi-Layer Perceptron (Neural Network)
model = MLPClassifier(
    # Hidden layer sizes: (8, 4) means 8 neurons in first layer, 4 in second
    hidden_layer_sizes=(8, 4),
    
    # Activation function: ReLU (Rectified Linear Unit)
    activation='relu',
    
    # Solver: Adam optimizer (adaptive learning rate)
    solver='adam',
    
    # Maximum iterations (epochs)
    max_iter=500,
    
    # Random state for reproducibility
    random_state=42,
    
    # Show training progress
    verbose=True
)

print("\n📐 Model Architecture:")
print("   Input Layer: 5 neurons (features)")
print("   Hidden Layer 1: 8 neurons (ReLU activation)")
print("   Hidden Layer 2: 4 neurons (ReLU activation)")
print("   Output Layer: 1 neuron (probability)")

# Step 4: Train the Model
print("\n🎓 Step 4: Training model...")
print("This may take a minute...\n")

# Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
print("\n📊 Step 5: Evaluating model...")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"✅ Training Score: {model.score(X_train, y_train)*100:.2f}%")

# Detailed classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No-Show', 'Show-Up']))

# Step 6: Save the Model and Scaler
print("\n💾 Step 6: Saving model and scaler...")

# Save model
joblib.dump(model, 'saved_model.pkl')
print("✅ Model saved as 'saved_model.pkl'")

# Save scaler (needed for preprocessing new data)
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as 'scaler.pkl'")

# Step 7: Test Prediction
print("\n🧪 Step 7: Testing prediction...")

# Create a sample patient
sample_patient = np.array([[
    35,  # Age: 35
    1,   # Procedure: root canal
    0,   # Previous no-show: no
    2,   # Visit day: Wednesday
    1    # Time slot: morning
]])

# Normalize using the same scaler
sample_scaled = scaler.transform(sample_patient)

# Make prediction (returns probability for each class)
prediction_proba = model.predict_proba(sample_scaled)
show_up_probability = prediction_proba[0][1]  # Probability of showing up
risk_percentage = (1 - show_up_probability) * 100  # Convert to no-show risk

print(f"\n📋 Sample Patient:")
print(f"   Age: 35, Procedure: Root Canal, Previous No-Show: No")
print(f"   Predicted Show-Up Probability: {show_up_probability*100:.1f}%")
print(f"   Predicted No-Show Risk: {risk_percentage:.1f}%")

# Training summary
print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\n📝 Summary:")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Test accuracy: {test_accuracy*100:.2f}%")
print(f"   • Model saved: saved_model.pkl")
print(f"   • Scaler saved: scaler.pkl")
print(f"   • Dataset saved: dataset.csv")
print("\n✅ You can now start the backend server:")
print("   uvicorn main:app --reload")
print("\n" + "=" * 60)
