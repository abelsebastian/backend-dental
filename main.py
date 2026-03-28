"""
FastAPI Backend Server
AI-powered dental appointment prediction system with SQLite database
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from datetime import datetime
import os
import random
from textblob import TextBlob
from sqlalchemy.orm import Session
from typing import Optional, List
from sentiment_engine import analyze_full as sentiment_analyze_full

from auth_endpoints import router as auth_router
from database import init_db, get_db, AppointmentDB, SentimentLogDB, UserDB, WaitlistDB, NotificationLogDB, DoctorDB
from auth import seed_demo_users

app = FastAPI(
    title="Smart DentalOps API",
    description="AI-powered dental appointment prediction system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

# ── Startup: init DB and seed demo users ──────────────────────────────────────
@app.on_event("startup")
def startup():
    init_db()
    from database import SessionLocal
    db = SessionLocal()
    try:
        seed_demo_users(db)
        seed_demo_doctors(db)
    finally:
        db.close()
    print("✅ Database initialized, demo users and doctors seeded")

# Load the trained ANN model and scaler
# This happens once when server starts, not on every request
MODEL_PATH = "saved_model.pkl"
SCALER_PATH = "scaler.pkl"
model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and scaler loaded successfully!")
    else:
        print("⚠️ Model or scaler file not found. Please run model.py first to train the model.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ============================================================================
# NEW MODULE: Scheduling Recommendation Engine
# ============================================================================

def get_scheduling_recommendation(risk_percentage, procedure):
    """
    Intelligent Scheduling Recommendation Engine
    
    This function takes the ANN prediction risk and recommends the best
    appointment slot strategy based on simple business rules.
    
    Args:
        risk_percentage (float): No-show risk from ANN (0-100)
        procedure (str): Type of dental procedure
    
    Returns:
        dict: Contains slot time, type, and reasoning
    
    Business Rules:
        - Risk < 40%: Standard Slot (preferred times, reliable patient)
        - Risk 40-70%: Confirmation Slot (requires confirmation call)
        - Risk > 70%: Backup Slot (overbook or alternate time)
    """
    
    # Rule 1: Low Risk (< 40%) - Standard Slot
    if risk_percentage < 40:
        return {
            "slot": "10:30 AM",
            "type": "Standard Slot",
            "reason": "Low risk patient - reliable attendance expected"
        }
    
    # Rule 2: Medium Risk (40-70%) - Confirmation Slot
    elif risk_percentage >= 40 and risk_percentage <= 70:
        return {
            "slot": "2:00 PM",
            "type": "Confirmation Slot",
            "reason": "Medium risk - requires confirmation call 24hrs before"
        }
    
    # Rule 3: High Risk (> 70%) - Backup Slot
    else:
        return {
            "slot": "4:30 PM",
            "type": "Backup Slot",
            "reason": "High risk - consider overbooking or waitlist strategy"
        }

# ============================================================================

# Define data structure for incoming requests
# Pydantic automatically validates the data
class PatientData(BaseModel):
    name: str
    age: int
    procedure: str  # "cleaning", "root canal", or "extraction"
    previousNoShow: bool

# Define data structure for response
class PredictionResponse(BaseModel):
    risk: str  # e.g., "45%"
    duration: str  # e.g., "30 min"
    slot: str  # e.g., "10:30 AM"
    slotType: str  # NEW: "Standard", "Confirmation", or "Backup"
    slotReason: str  # NEW: Explanation for the slot recommendation

# Root endpoint - just to check if server is running
@app.get("/")
def read_root():
    """
    Health check endpoint
    Returns basic info about the API
    """
    return {
        "message": "Smart DentalOps API is running!",
        "status": "active",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

# Main prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_appointment(patient: PatientData):
    """
    Predict no-show risk for a patient appointment
    
    Args:
        patient: PatientData object with name, age, procedure, previousNoShow
    
    Returns:
        PredictionResponse with risk percentage, duration, and suggested slot
    """
    
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please train the model first by running model.py"
        )
    
    try:
        # Step 1: Preprocess input data
        # Convert procedure type to numeric value
        procedure_map = {
            "cleaning": 0,
            "root canal": 1,
            "extraction": 2
        }
        procedure_encoded = procedure_map.get(patient.procedure.lower(), 0)
        
        # Convert previous no-show to numeric (0 or 1)
        previous_no_show_encoded = 1 if patient.previousNoShow else 0
        
        # Get current day of week (0=Monday, 6=Sunday)
        current_day = datetime.now().weekday()
        
        # Assume morning slot (can be enhanced later)
        time_slot = 1  # 0=early morning, 1=morning, 2=afternoon, 3=evening
        
        # Step 2: Create input array for model
        # Order: [age, procedure, previousNoShow, visitDay, timeSlot]
        input_data = np.array([[
            patient.age,
            procedure_encoded,
            previous_no_show_encoded,
            current_day,
            time_slot
        ]])
        
        # Step 3: Normalize using the saved scaler
        input_scaled = scaler.transform(input_data)
        
        # Step 4: Make prediction using the neural network
        # predict_proba returns [probability_no_show, probability_show_up]
        prediction_proba = model.predict_proba(input_scaled)
        show_up_probability = float(prediction_proba[0][1])  # Probability of showing up
        risk_percentage = round((1 - show_up_probability) * 100, 1)  # Convert to no-show risk
        
        # Step 5: Apply business logic
        
        # Determine appointment duration based on procedure type
        duration_map = {
            "cleaning": "20 min",
            "root canal": "45 min",
            "extraction": "30 min"
        }
        duration = duration_map.get(patient.procedure.lower(), "30 min")
        
        # NEW MODULE: Scheduling Recommendation Engine
        # This uses the ANN risk output to intelligently recommend appointment slots
        slot_recommendation = get_scheduling_recommendation(risk_percentage, patient.procedure)
        slot = slot_recommendation["slot"]
        slot_type = slot_recommendation["type"]
        slot_reason = slot_recommendation["reason"]
        
        # Step 6: Prepare and return response (with new scheduling fields)
        response = PredictionResponse(
            risk=f"{risk_percentage}%",
            duration=duration,
            slot=slot,
            slotType=slot_type,
            slotReason=slot_reason
        )
        
        # Log prediction for debugging (in production, save to database)
        print(f"📊 Prediction for {patient.name}: Risk={risk_percentage}%, Duration={duration}")
        
        return response
        
    except Exception as e:
        # Handle any errors during prediction
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

# Additional endpoint to get model info (useful for debugging)
@app.get("/model-info")
def get_model_info():
    """
    Returns information about the loaded model
    """
    if model is None:
        return {"status": "Model not loaded"}
    
    return {
        "status": "Model loaded",
        "model_type": "MLPClassifier (Multi-Layer Perceptron)",
        "hidden_layers": str(model.hidden_layer_sizes),
        "activation": model.activation,
        "solver": model.solver,
        "n_features": model.n_features_in_,
        "n_outputs": model.n_outputs_
    }

# ============================================================================
# PHASE 1: Real-Time Synthetic Data Generation
# ============================================================================

def generate_synthetic_appointment():
    """
    Generate a single synthetic appointment with realistic random variations.
    
    This simulates real-time appointment data by creating random patient records
    with varied attributes. Used for live dashboard updates.
    
    Returns:
        dict: Synthetic appointment data with all required fields
    """
    
    # Sample patient names for variety
    first_names = ["John", "Sarah", "Michael", "Emma", "David", "Lisa", "James", "Maria", "Robert", "Jennifer"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    
    # Generate random patient data
    patient_id = random.randint(1000, 9999)
    patient_name = f"{random.choice(first_names)} {random.choice(last_names)}"
    age = random.randint(18, 80)
    
    # Procedure types with realistic distribution
    procedures = ["Cleaning", "Root Canal", "Extraction", "Filling", "Checkup"]
    procedure_weights = [0.4, 0.15, 0.1, 0.25, 0.1]  # Cleaning is most common
    procedure_type = random.choices(procedures, weights=procedure_weights)[0]
    
    # Generate risk score (0-100) with realistic distribution
    # Most patients have low-medium risk
    risk_score = round(random.triangular(10, 90, 35), 1)
    
    # Determine slot type based on risk score
    if risk_score < 40:
        slot_type = "Standard Slot"
        slot_time = random.choice(["9:00 AM", "10:30 AM", "11:00 AM"])
    elif risk_score <= 70:
        slot_type = "Confirmation Slot"
        slot_time = random.choice(["1:00 PM", "2:00 PM", "3:00 PM"])
    else:
        slot_type = "Backup Slot"
        slot_time = random.choice(["4:00 PM", "4:30 PM", "5:00 PM"])
    
    # Appointment status with realistic distribution
    statuses = ["Scheduled", "Confirmed", "In Progress", "Completed", "Cancelled"]
    status_weights = [0.3, 0.25, 0.15, 0.25, 0.05]
    status = random.choices(statuses, weights=status_weights)[0]
    
    # Assign dentist and chair
    dentists = ["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown"]
    dentist = random.choice(dentists)
    
    chairs = ["Chair 1", "Chair 2", "Chair 3", "Chair 4", "Chair 5"]
    chair = random.choice(chairs)
    
    # Duration based on procedure type
    duration_map = {
        "Cleaning": "20 min",
        "Root Canal": "45 min",
        "Extraction": "30 min",
        "Filling": "25 min",
        "Checkup": "15 min"
    }
    duration = duration_map.get(procedure_type, "30 min")
    
    return {
        "patientId": patient_id,
        "patientName": patient_name,
        "age": age,
        "procedureType": procedure_type,
        "riskScore": risk_score,
        "slotType": slot_type,
        "slotTime": slot_time,
        "status": status,
        "dentist": dentist,
        "chair": chair,
        "duration": duration,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/live-dashboard-data")
async def get_live_dashboard_data():
    """
    PHASE 1: Real-Time Synthetic Data Endpoint
    
    Generates synthetic appointment data for live dashboard updates.
    Frontend polls this endpoint every 5 seconds to simulate real-time updates.
    
    Returns:
        dict: Contains list of synthetic appointments and metadata
    """
    
    try:
        # Generate 5-10 random appointments for the dashboard
        num_appointments = random.randint(5, 10)
        appointments = [generate_synthetic_appointment() for _ in range(num_appointments)]
        
        # Calculate summary statistics
        total_appointments = len(appointments)
        high_risk_count = sum(1 for apt in appointments if apt["riskScore"] > 70)
        avg_risk = round(sum(apt["riskScore"] for apt in appointments) / total_appointments, 1)
        
        # Chair utilization (percentage of chairs in use)
        in_progress = sum(1 for apt in appointments if apt["status"] == "In Progress")
        chair_utilization = round((in_progress / 5) * 100, 1)  # 5 total chairs
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "appointments": appointments,
            "summary": {
                "totalAppointments": total_appointments,
                "highRiskCount": high_risk_count,
                "averageRisk": avg_risk,
                "chairUtilization": chair_utilization
            }
        }
        
    except Exception as e:
        print(f"❌ Error generating live data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating live dashboard data: {str(e)}"
        )

# ============================================================================
# PHASE 2: Sentiment Analysis
# ============================================================================

class SentimentRequest(BaseModel):
    """
    Request model for sentiment analysis
    Contains the patient message to analyze
    """
    message: str
    currentRisk: float  # Current risk score from ANN (0-100)

class SentimentResponse(BaseModel):
    """
    Response model for sentiment analysis
    Contains sentiment category, polarity score, and adjusted risk
    """
    sentiment: str  # "positive", "neutral", or "negative"
    polarity: float  # -1.0 to 1.0 (TextBlob polarity score)
    subjectivity: float  # 0.0 to 1.0 (how subjective vs objective)
    adjustedRisk: float  # Risk after sentiment adjustment
    riskChange: float  # How much risk changed
    explanation: str  # Human-readable explanation

def analyze_sentiment(message: str):
    """
    Analyze sentiment of patient message using TextBlob
    
    TextBlob provides:
    - Polarity: -1 (negative) to +1 (positive)
    - Subjectivity: 0 (objective) to 1 (subjective)
    
    Args:
        message (str): Patient message text
    
    Returns:
        dict: Sentiment analysis results
    """
    
    # Create TextBlob object
    blob = TextBlob(message)
    
    # Get sentiment scores
    polarity = blob.sentiment.polarity  # -1 to +1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1
    
    # Categorize sentiment based on polarity
    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3)
    }

def adjust_risk_by_sentiment(current_risk: float, sentiment: str, polarity: float):
    """
    Adjust no-show risk based on sentiment analysis
    
    Business Rules:
    - Positive sentiment: Keep risk unchanged (patient is engaged)
    - Neutral sentiment: Keep risk unchanged
    - Negative sentiment: Add 15% to risk (patient may be frustrated/unhappy)
    
    Args:
        current_risk (float): Current risk score (0-100)
        sentiment (str): Sentiment category
        polarity (float): Polarity score from TextBlob
    
    Returns:
        dict: Adjusted risk and explanation
    """
    
    risk_change = 0.0
    explanation = ""
    
    if sentiment == "positive":
        # Positive sentiment - no change
        risk_change = 0.0
        explanation = "Positive sentiment detected. Patient appears engaged and satisfied. Risk unchanged."
    
    elif sentiment == "neutral":
        # Neutral sentiment - no change
        risk_change = 0.0
        explanation = "Neutral sentiment detected. Standard communication. Risk unchanged."
    
    elif sentiment == "negative":
        # Negative sentiment - increase risk by 15%
        risk_change = 15.0
        explanation = "Negative sentiment detected. Patient may be frustrated or unhappy. Risk increased by 15%."
    
    # Calculate adjusted risk (cap at 100%)
    adjusted_risk = min(current_risk + risk_change, 100.0)
    
    return {
        "adjustedRisk": round(adjusted_risk, 1),
        "riskChange": risk_change,
        "explanation": explanation
    }

@app.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_patient_sentiment(request: SentimentRequest):
    """
    PHASE 2 (Enhanced): Multi-layer Sentiment Analysis
    
    Uses VADER + dental domain lexicon + negation handling + emotion classification.
    Much more accurate than basic TextBlob for medical/dental context.
    """
    try:
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        if request.currentRisk < 0 or request.currentRisk > 100:
            raise HTTPException(status_code=400, detail="Current risk must be between 0 and 100")

        # Run full multi-layer analysis
        result = sentiment_analyze_full(request.message, request.currentRisk)

        response = SentimentResponse(
            sentiment=result["sentiment"],
            polarity=result["polarity"],
            subjectivity=result["subjectivity"],
            adjustedRisk=result["adjustedRisk"],
            riskChange=result["riskChange"],
            explanation=result["explanation"],
        )

        print(f"📝 Enhanced Sentiment: '{request.message[:50]}' → {result['sentiment']} "
              f"(VADER:{result['vaderScore']:.2f}, Domain:{result['domainScore']:.2f}, "
              f"Fused:{result['polarity']:.2f}) | Emotions: {result['emotionList']}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

# ============================================================================
# PHASE 3: Agentic Communication Analysis (Intent Detection)
# ============================================================================

class IntentRequest(BaseModel):
    """
    Request model for intent detection
    Contains the patient message to analyze
    """
    message: str
    currentRisk: float  # Current risk score (0-100)
    sentiment: str = None  # Optional: sentiment from Phase 2

class IntentResponse(BaseModel):
    """
    Response model for intent detection
    Contains detected intent and adjusted risk
    """
    intent: str  # "Confirmation", "Delay", "Cancellation", or "Unknown"
    confidence: str  # "High", "Medium", or "Low"
    adjustedRisk: float  # Risk after intent adjustment
    riskChange: float  # How much risk changed
    explanation: str  # Human-readable explanation
    keywords: list  # Keywords that triggered the intent

def detect_intent(message: str):
    """
    Detect patient intent from message using keyword matching
    
    Simple rule-based intent detection:
    - Cancellation: Keywords like "cancel", "can't make it", "won't be there"
    - Delay: Keywords like "late", "running behind", "reschedule"
    - Confirmation: Keywords like "confirm", "yes", "will be there"
    - Unknown: No clear intent detected
    
    Args:
        message (str): Patient message text
    
    Returns:
        dict: Intent detection results
    """
    
    # Convert to lowercase for matching
    message_lower = message.lower()
    
    # Define keyword patterns for each intent
    cancellation_keywords = [
        "cancel", "can't make", "cannot make", "won't be there",
        "not coming", "skip", "miss", "unable to attend",
        "have to cancel", "need to cancel", "want to cancel"
    ]
    
    delay_keywords = [
        "late", "running late", "running behind", "delay",
        "reschedule", "change time", "different time",
        "move appointment", "postpone", "push back"
    ]
    
    confirmation_keywords = [
        "confirm", "yes", "will be there", "see you",
        "looking forward", "on my way", "be there",
        "ready", "prepared", "excited"
    ]
    
    # Check for each intent (priority: cancellation > delay > confirmation)
    detected_keywords = []
    intent = "Unknown"
    confidence = "Low"
    
    # Check for cancellation (highest priority)
    for keyword in cancellation_keywords:
        if keyword in message_lower:
            detected_keywords.append(keyword)
            intent = "Cancellation"
            confidence = "High"
            break
    
    # Check for delay if no cancellation
    if intent == "Unknown":
        for keyword in delay_keywords:
            if keyword in message_lower:
                detected_keywords.append(keyword)
                intent = "Delay"
                confidence = "High"
                break
    
    # Check for confirmation if no cancellation or delay
    if intent == "Unknown":
        for keyword in confirmation_keywords:
            if keyword in message_lower:
                detected_keywords.append(keyword)
                intent = "Confirmation"
                confidence = "High"
                break
    
    # If still unknown, check for weaker signals
    if intent == "Unknown":
        if any(word in message_lower for word in ["maybe", "not sure", "uncertain"]):
            intent = "Delay"
            confidence = "Low"
            detected_keywords.append("uncertainty detected")
        elif any(word in message_lower for word in ["ok", "okay", "fine", "sure"]):
            intent = "Confirmation"
            confidence = "Medium"
            detected_keywords.append("weak confirmation")
    
    return {
        "intent": intent,
        "confidence": confidence,
        "keywords": detected_keywords
    }

def adjust_risk_by_intent(current_risk: float, intent: str, confidence: str):
    """
    Adjust no-show risk based on detected intent
    
    Business Rules:
    - Confirmation: Reduce risk by 10% (patient committed to attending)
    - Delay: Add 5% to risk (patient may have scheduling conflicts)
    - Cancellation: Add 25% to risk (strong signal of no-show)
    - Unknown: No change
    
    Confidence affects adjustment:
    - High confidence: Full adjustment
    - Medium confidence: 50% adjustment
    - Low confidence: 25% adjustment
    
    Args:
        current_risk (float): Current risk score (0-100)
        intent (str): Detected intent
        confidence (str): Confidence level
    
    Returns:
        dict: Adjusted risk and explanation
    """
    
    # Base risk changes by intent
    base_changes = {
        "Confirmation": -10.0,
        "Delay": 5.0,
        "Cancellation": 25.0,
        "Unknown": 0.0
    }
    
    # Confidence multipliers
    confidence_multipliers = {
        "High": 1.0,
        "Medium": 0.5,
        "Low": 0.25
    }
    
    # Calculate risk change
    base_change = base_changes.get(intent, 0.0)
    multiplier = confidence_multipliers.get(confidence, 1.0)
    risk_change = base_change * multiplier
    
    # Calculate adjusted risk (clamp between 0 and 100)
    adjusted_risk = max(0.0, min(current_risk + risk_change, 100.0))
    
    # Generate explanation
    if intent == "Confirmation":
        explanation = f"{confidence} confidence confirmation detected. Patient committed to attending. Risk decreased by {abs(risk_change):.1f}%."
    elif intent == "Delay":
        explanation = f"{confidence} confidence delay/reschedule intent detected. Patient may have scheduling conflicts. Risk increased by {risk_change:.1f}%."
    elif intent == "Cancellation":
        explanation = f"{confidence} confidence cancellation intent detected. Strong signal of potential no-show. Risk increased by {risk_change:.1f}%."
    else:
        explanation = "No clear intent detected. Risk unchanged."
    
    return {
        "adjustedRisk": round(adjusted_risk, 1),
        "riskChange": round(risk_change, 1),
        "explanation": explanation
    }

@app.post("/detect-intent", response_model=IntentResponse)
async def detect_patient_intent(request: IntentRequest):
    """
    PHASE 3 (Enhanced): Multi-confidence Intent Detection
    
    Uses pattern matching with High/Medium/Low confidence levels,
    detects multiple intents simultaneously, and applies weighted risk adjustment.
    """
    try:
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        if request.currentRisk < 0 or request.currentRisk > 100:
            raise HTTPException(status_code=400, detail="Current risk must be between 0 and 100")

        # Use the full engine for intent + risk
        result = sentiment_analyze_full(request.message, request.currentRisk)

        response = IntentResponse(
            intent=result["intent"],
            confidence=result["intentConfidence"],
            adjustedRisk=result["adjustedRisk"],
            riskChange=result["riskChange"],
            explanation=result["explanation"],
            keywords=result["intentKeywords"],
        )

        print(f"🎯 Enhanced Intent: '{request.message[:50]}' → {result['intent']} "
              f"({result['intentConfidence']}) | Risk: {request.currentRisk}% → {result['adjustedRisk']}%")

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in intent detection: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting intent: {str(e)}")

# ============================================================================

# ============================================================================
# ENHANCED SENTIMENT ANALYSIS ENDPOINT (full response)
# ============================================================================

class FullSentimentRequest(BaseModel):
    message: str
    currentRisk: float = 50.0

@app.post("/analyze-full")
async def analyze_full_sentiment(request: FullSentimentRequest):
    """
    Full multi-layer sentiment analysis returning all signals:
    - VADER score, domain score, fused polarity
    - Emotion classification (anxiety, satisfaction, frustration, trust, urgency, pain)
    - Intent detection with all detected intents and confidence
    - Risk adjustment breakdown
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        result = sentiment_analyze_full(request.message, request.currentRisk)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DATABASE-BACKED ENDPOINTS
# ============================================================================

# ── Appointment models ────────────────────────────────────────────────────────

class AppointmentCreate(BaseModel):
    patient_name: str
    age: int = 30
    procedure_type: str = "Cleaning"
    risk_score: float = 0.0
    slot_type: str = "Standard Slot"
    slot_time: str = "10:00 AM"
    status: str = "Scheduled"
    dentist: str = "Dr. Smith"
    chair: str = "Chair 1"
    duration: str = "30 min"
    notes: str = ""

class AppointmentUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    dentist: Optional[str] = None
    slot_time: Optional[str] = None

def apt_to_dict(a: AppointmentDB) -> dict:
    return {
        "id": a.id, "patientName": a.patient_name, "patientId": a.patient_id,
        "age": a.age, "procedureType": a.procedure_type, "riskScore": a.risk_score,
        "slotType": a.slot_type, "slotTime": a.slot_time, "status": a.status,
        "dentist": a.dentist, "chair": a.chair, "duration": a.duration,
        "notes": a.notes, "createdAt": a.created_at.isoformat() if a.created_at else None,
    }

# ── GET /appointments ─────────────────────────────────────────────────────────

@app.get("/appointments")
async def get_appointments(
    status: Optional[str] = None,
    dentist: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all appointments from DB with optional filters"""
    q = db.query(AppointmentDB)
    if status:
        q = q.filter(AppointmentDB.status == status)
    if dentist:
        q = q.filter(AppointmentDB.dentist == dentist)
    appointments = q.order_by(AppointmentDB.created_at.desc()).limit(limit).all()
    return {"appointments": [apt_to_dict(a) for a in appointments], "total": len(appointments)}

# ── POST /appointments ────────────────────────────────────────────────────────

@app.post("/appointments")
async def create_appointment(data: AppointmentCreate, db: Session = Depends(get_db)):
    """Create a new appointment in DB"""
    patient_id = f"#SDO-{random.randint(1000, 9999)}"
    apt = AppointmentDB(
        patient_name=data.patient_name,
        patient_id=patient_id,
        age=data.age,
        procedure_type=data.procedure_type,
        risk_score=data.risk_score,
        slot_type=data.slot_type,
        slot_time=data.slot_time,
        status=data.status,
        dentist=data.dentist,
        chair=data.chair,
        duration=data.duration,
        notes=data.notes,
        created_at=datetime.now(),
    )
    db.add(apt)
    db.commit()
    db.refresh(apt)
    return apt_to_dict(apt)

# ── PATCH /appointments/{id} ──────────────────────────────────────────────────

@app.patch("/appointments/{apt_id}")
async def update_appointment(apt_id: int, data: AppointmentUpdate, db: Session = Depends(get_db)):
    """Update appointment status or notes"""
    apt = db.query(AppointmentDB).filter(AppointmentDB.id == apt_id).first()
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    if data.status is not None:
        apt.status = data.status
    if data.notes is not None:
        apt.notes = data.notes
    if data.dentist is not None:
        apt.dentist = data.dentist
    if data.slot_time is not None:
        apt.slot_time = data.slot_time
    apt.updated_at = datetime.now()
    db.commit()
    db.refresh(apt)
    return apt_to_dict(apt)

# ── DELETE /appointments/{id} ─────────────────────────────────────────────────

@app.delete("/appointments/{apt_id}")
async def delete_appointment(apt_id: int, db: Session = Depends(get_db)):
    """Delete an appointment"""
    apt = db.query(AppointmentDB).filter(AppointmentDB.id == apt_id).first()
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    db.delete(apt)
    db.commit()
    return {"message": "Appointment deleted"}

# ── GET /analytics/summary ────────────────────────────────────────────────────

@app.get("/analytics/summary")
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Real analytics from DB"""
    all_apts = db.query(AppointmentDB).all()
    total = len(all_apts)

    if total == 0:
        # Return synthetic data if DB is empty
        return {
            "totalAppointments": 1284,
            "todayAppointments": 24,
            "highRiskCount": 12,
            "averageRisk": 35.2,
            "chairUtilization": 72,
            "noShowRate": 4.8,
            "monthlyRevenue": 42500,
            "source": "synthetic"
        }

    high_risk = sum(1 for a in all_apts if a.risk_score > 70)
    avg_risk = round(sum(a.risk_score for a in all_apts) / total, 1)
    in_progress = sum(1 for a in all_apts if a.status == "In Progress")
    chair_util = round((in_progress / max(5, 1)) * 100, 1)
    cancelled = sum(1 for a in all_apts if a.status == "Cancelled")
    no_show_rate = round((cancelled / total) * 100, 1) if total > 0 else 0

    # Dentist workload
    dentist_counts = {}
    for a in all_apts:
        dentist_counts[a.dentist] = dentist_counts.get(a.dentist, 0) + 1

    # Procedure breakdown
    procedure_counts = {}
    for a in all_apts:
        procedure_counts[a.procedure_type] = procedure_counts.get(a.procedure_type, 0) + 1

    return {
        "totalAppointments": total,
        "todayAppointments": total,
        "highRiskCount": high_risk,
        "averageRisk": avg_risk,
        "chairUtilization": chair_util,
        "noShowRate": no_show_rate,
        "monthlyRevenue": total * 150,
        "dentistWorkload": dentist_counts,
        "procedureBreakdown": procedure_counts,
        "source": "database"
    }

# ── POST /sentiment-log ───────────────────────────────────────────────────────

@app.post("/sentiment-log")
async def log_sentiment(
    patient_name: str,
    message: str,
    sentiment: str,
    polarity: float,
    subjectivity: float,
    intent: str,
    confidence: str,
    risk_before: float,
    risk_after: float,
    db: Session = Depends(get_db)
):
    """Save sentiment analysis result to DB"""
    log = SentimentLogDB(
        patient_name=patient_name,
        message=message,
        sentiment=sentiment,
        polarity=polarity,
        subjectivity=subjectivity,
        intent=intent,
        confidence=confidence,
        risk_before=risk_before,
        risk_after=risk_after,
        created_at=datetime.now(),
    )
    db.add(log)
    db.commit()
    return {"message": "Logged successfully"}

# ── GET /sentiment-logs ───────────────────────────────────────────────────────

@app.get("/sentiment-logs")
async def get_sentiment_logs(limit: int = 20, db: Session = Depends(get_db)):
    """Get sentiment analysis history from DB"""
    logs = db.query(SentimentLogDB).order_by(SentimentLogDB.created_at.desc()).limit(limit).all()
    return [{
        "id": l.id, "patientName": l.patient_name, "message": l.message[:100],
        "sentiment": l.sentiment, "polarity": l.polarity, "intent": l.intent,
        "confidence": l.confidence, "riskBefore": l.risk_before, "riskAfter": l.risk_after,
        "createdAt": l.created_at.isoformat() if l.created_at else None,
    } for l in logs]

# ── GET /db-stats ─────────────────────────────────────────────────────────────

@app.get("/db-stats")
async def get_db_stats(db: Session = Depends(get_db)):
    """Quick DB health check"""
    return {
        "users": db.query(UserDB).count(),
        "appointments": db.query(AppointmentDB).count(),
        "sentiment_logs": db.query(SentimentLogDB).count(),
        "database": "SQLite (persistent)",
    }

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================================================
# DOCTOR SEEDING
# ============================================================================

def seed_demo_doctors(db: Session):
    """Seed demo doctors if none exist"""
    if db.query(DoctorDB).count() == 0:
        doctors = [
            DoctorDB(name="Dr. Julian Vance", specialty="General Dentistry", email="vance@dentalops.com", phone="+1-555-0101"),
            DoctorDB(name="Dr. Sarah Chen", specialty="Orthodontics", email="chen@dentalops.com", phone="+1-555-0102"),
            DoctorDB(name="Dr. Marcus Thorne", specialty="Endodontics", email="thorne@dentalops.com", phone="+1-555-0103"),
            DoctorDB(name="Dr. Elena Rodriguez", specialty="Oral Surgery", email="rodriguez@dentalops.com", phone="+1-555-0104"),
        ]
        for d in doctors:
            db.add(d)
        db.commit()
        print("✅ Demo doctors seeded")


# ============================================================================
# DOCTORS ENDPOINTS
# ============================================================================

@app.get("/doctors")
async def get_doctors(specialty: Optional[str] = None, db: Session = Depends(get_db)):
    """List all active doctors with optional specialty filter"""
    q = db.query(DoctorDB).filter(DoctorDB.is_active == True)
    if specialty:
        q = q.filter(DoctorDB.specialty.ilike(f"%{specialty}%"))
    doctors = q.all()
    return [{
        "id": d.id, "name": d.name, "specialty": d.specialty,
        "email": d.email, "phone": d.phone,
        "availableDays": d.available_days.split(","),
        "startTime": d.start_time, "endTime": d.end_time,
        "slotDurationMin": d.slot_duration_min,
    } for d in doctors]


@app.get("/doctors/{doctor_id}/slots")
async def get_doctor_slots(doctor_id: int, date: Optional[str] = None, db: Session = Depends(get_db)):
    """Get available slots for a doctor on a given date"""
    doctor = db.query(DoctorDB).filter(DoctorDB.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    # Generate time slots based on doctor schedule
    from datetime import time as dtime
    start_h, start_m = map(int, doctor.start_time.split(":"))
    end_h, end_m = map(int, doctor.end_time.split(":"))
    duration = doctor.slot_duration_min

    slots = []
    current_min = start_h * 60 + start_m
    end_min = end_h * 60 + end_m

    # Get booked slots for this date
    booked_times = set()
    if date:
        booked = db.query(AppointmentDB).filter(
            AppointmentDB.dentist == doctor.name,
            AppointmentDB.appointment_date == date,
            AppointmentDB.status.notin_(["Cancelled", "No-Show"])
        ).all()
        booked_times = {a.slot_time for a in booked}

    while current_min + duration <= end_min:
        h = current_min // 60
        m = current_min % 60
        period = "AM" if h < 12 else "PM"
        display_h = h if h <= 12 else h - 12
        if display_h == 0:
            display_h = 12
        slot_str = f"{display_h}:{m:02d} {period}"
        slots.append({
            "time": slot_str,
            "available": slot_str not in booked_times,
            "duration": f"{duration} min"
        })
        current_min += duration

    return {"doctor": doctor.name, "date": date or "any", "slots": slots}


# ============================================================================
# SMART RESCHEDULING ENGINE
# ============================================================================

class RescheduleRequest(BaseModel):
    appointment_id: int
    preferred_time: str = "any"   # morning / afternoon / evening / any
    preferred_date_from: Optional[str] = None
    preferred_date_to: Optional[str] = None

@app.post("/appointments/{apt_id}/reschedule-options")
async def get_reschedule_options(apt_id: int, request: RescheduleRequest, db: Session = Depends(get_db)):
    """
    Smart Rescheduling: Returns top 3 alternative slots ranked by:
    - Patient preference match
    - Predicted attendance probability (inverse of risk)
    - Proximity to original appointment
    """
    apt = db.query(AppointmentDB).filter(AppointmentDB.id == apt_id).first()
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    # Generate candidate slots (next 14 days)
    from datetime import date, timedelta
    today = date.today()
    candidates = []

    time_preferences = {
        "morning": ["9:00 AM", "9:30 AM", "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM"],
        "afternoon": ["1:00 PM", "1:30 PM", "2:00 PM", "2:30 PM", "3:00 PM", "3:30 PM"],
        "evening": ["4:00 PM", "4:30 PM", "5:00 PM"],
        "any": ["9:00 AM", "10:00 AM", "11:00 AM", "2:00 PM", "3:00 PM", "4:00 PM"],
    }
    preferred_slots = time_preferences.get(request.preferred_time, time_preferences["any"])

    for day_offset in range(1, 15):
        candidate_date = today + timedelta(days=day_offset)
        date_str = candidate_date.strftime("%Y-%m-%d")
        day_name = candidate_date.strftime("%A")

        # Skip weekends for standard scheduling
        if day_name in ["Saturday", "Sunday"]:
            continue

        for slot_time in preferred_slots:
            # Check if slot is already booked
            existing = db.query(AppointmentDB).filter(
                AppointmentDB.dentist == apt.dentist,
                AppointmentDB.appointment_date == date_str,
                AppointmentDB.slot_time == slot_time,
                AppointmentDB.status.notin_(["Cancelled", "No-Show"])
            ).first()

            if not existing:
                # Score this slot
                pref_score = 1.0 if slot_time in preferred_slots else 0.5
                # Proximity score: closer dates score higher
                proximity_score = max(0, 1.0 - (day_offset / 14))
                # Attendance score: use inverse of patient's risk
                attendance_score = max(0, 1.0 - (apt.risk_score / 100))

                total_score = (pref_score * 0.4) + (attendance_score * 0.4) + (proximity_score * 0.2)

                candidates.append({
                    "date": date_str,
                    "dayName": day_name,
                    "displayDate": candidate_date.strftime("%B %d, %Y"),
                    "time": slot_time,
                    "doctor": apt.dentist,
                    "procedure": apt.procedure_type,
                    "score": round(total_score * 100),
                    "attendanceProbability": round(attendance_score * 100),
                    "label": f"{day_name}, {candidate_date.strftime('%B %d')} at {slot_time}"
                })

    # Sort by score and return top 3
    candidates.sort(key=lambda x: x["score"], reverse=True)
    top3 = candidates[:3]

    return {
        "originalAppointment": {
            "id": apt.id, "patient": apt.patient_name,
            "procedure": apt.procedure_type, "dentist": apt.dentist,
            "currentSlot": f"{apt.appointment_date} {apt.slot_time}"
        },
        "suggestedSlots": top3,
        "message": f"Found {len(top3)} alternative slots based on your preferences and predicted attendance."
    }


@app.post("/appointments/{apt_id}/confirm-reschedule")
async def confirm_reschedule(apt_id: int, new_date: str, new_time: str, db: Session = Depends(get_db)):
    """Apply the rescheduled slot to the appointment"""
    apt = db.query(AppointmentDB).filter(AppointmentDB.id == apt_id).first()
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    old_slot = f"{apt.appointment_date} {apt.slot_time}"
    apt.appointment_date = new_date
    apt.slot_time = new_time
    apt.status = "Scheduled"
    apt.confirmation_status = "Pending"
    apt.reminder_24h_sent = False
    apt.reminder_2h_sent = False
    apt.updated_at = datetime.now()
    db.commit()

    # Log notification
    notif = NotificationLogDB(
        appointment_id=apt_id,
        patient_name=apt.patient_name,
        patient_contact=apt.patient_email or apt.patient_phone,
        channel="System",
        notification_type="Rescheduled",
        message=f"Appointment rescheduled from {old_slot} to {new_date} {new_time}",
        status="Sent"
    )
    db.add(notif)
    db.commit()

    return {"message": "Appointment rescheduled successfully", "newSlot": f"{new_date} {new_time}"}


# ============================================================================
# RISK-BASED RESPONSE LOGIC
# ============================================================================

@app.post("/appointments/{apt_id}/apply-risk-response")
async def apply_risk_response(apt_id: int, db: Session = Depends(get_db)):
    """
    Apply smart risk-based response logic:
    - LOW (0-39%): Confirm instantly
    - MEDIUM (40-69%): Schedule reminders at T-24h and T-2h
    - HIGH (70%+): Request double confirmation + optional deposit
    """
    apt = db.query(AppointmentDB).filter(AppointmentDB.id == apt_id).first()
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    risk = apt.risk_score
    actions_taken = []

    if risk < 40:
        # LOW RISK: Confirm instantly
        apt.status = "Confirmed"
        apt.confirmation_status = "Confirmed"
        actions_taken.append("Appointment auto-confirmed (low risk)")
        notif_type = "AutoConfirmed"
        message = f"Hi {apt.patient_name}, your appointment with {apt.dentist} is confirmed. See you soon!"

    elif risk < 70:
        # MEDIUM RISK: Keep scheduled, flag for reminders
        apt.status = "Scheduled"
        actions_taken.append("Reminders scheduled for T-24h and T-2h")
        notif_type = "ReminderScheduled"
        message = f"Hi {apt.patient_name}, your appointment is confirmed. We'll send you a reminder 24 hours before."

    else:
        # HIGH RISK: Request double confirmation + deposit
        apt.status = "Scheduled"
        apt.deposit_required = True
        actions_taken.append("Double confirmation requested")
        actions_taken.append("Deposit requirement flagged")
        notif_type = "DoubleConfirmRequired"
        message = f"Hi {apt.patient_name}, please confirm your appointment with {apt.dentist}. Reply YES to confirm or NO to cancel."

    apt.updated_at = datetime.now()
    db.commit()

    # Log the notification
    notif = NotificationLogDB(
        appointment_id=apt_id,
        patient_name=apt.patient_name,
        patient_contact=apt.patient_email or apt.patient_phone,
        channel="SMS",
        notification_type=notif_type,
        message=message,
        status="Sent"
    )
    db.add(notif)
    db.commit()

    return {
        "appointmentId": apt_id,
        "riskScore": risk,
        "riskCategory": apt.risk_category,
        "actionsTaken": actions_taken,
        "notificationSent": message,
        "depositRequired": apt.deposit_required,
    }


@app.post("/appointments/{apt_id}/patient-confirm")
async def patient_confirm_appointment(apt_id: int, confirmed: bool, db: Session = Depends(get_db)):
    """Patient confirms or declines their appointment"""
    apt = db.query(AppointmentDB).filter(AppointmentDB.id == apt_id).first()
    if not apt:
        raise HTTPException(status_code=404, detail="Appointment not found")

    if confirmed:
        apt.confirmation_status = "Confirmed"
        apt.status = "Confirmed"
        message = "Appointment confirmed. See you soon!"
    else:
        apt.confirmation_status = "Declined"
        apt.status = "Cancelled"
        message = "Appointment cancelled. Slot released to waitlist."
        # Notify waitlist
        _notify_waitlist_for_slot(apt.dentist, apt.slot_time, apt.procedure_type, db)

    apt.updated_at = datetime.now()
    db.commit()

    return {"message": message, "status": apt.status, "confirmationStatus": apt.confirmation_status}


def _notify_waitlist_for_slot(dentist: str, slot_time: str, procedure_type: str, db: Session):
    """Internal: notify first waitlist patient when a slot opens"""
    waitlist_entry = db.query(WaitlistDB).filter(
        WaitlistDB.status == "Waiting",
        WaitlistDB.procedure_type == procedure_type
    ).order_by(WaitlistDB.created_at).first()

    if waitlist_entry:
        waitlist_entry.status = "Notified"
        waitlist_entry.notified_at = datetime.now()
        db.commit()

        notif = NotificationLogDB(
            patient_name=waitlist_entry.patient_name,
            patient_contact=waitlist_entry.patient_email or waitlist_entry.patient_phone,
            channel="SMS",
            notification_type="WaitlistAlert",
            message=f"Hi {waitlist_entry.patient_name}! A slot opened with {dentist} at {slot_time}. Reply BOOK to claim it within 15 minutes.",
            status="Sent"
        )
        db.add(notif)
        db.commit()


# ============================================================================
# WAITLIST ENDPOINTS
# ============================================================================

class WaitlistCreate(BaseModel):
    patient_name: str
    patient_email: str = ""
    patient_phone: str = ""
    procedure_type: str = "Cleaning"
    preferred_dentist: str = "Any"
    preferred_time: str = "any"
    preferred_date_from: str = ""
    preferred_date_to: str = ""

@app.post("/waitlist")
async def join_waitlist(data: WaitlistCreate, db: Session = Depends(get_db)):
    """Add patient to waitlist"""
    entry = WaitlistDB(
        patient_name=data.patient_name,
        patient_email=data.patient_email,
        patient_phone=data.patient_phone,
        procedure_type=data.procedure_type,
        preferred_dentist=data.preferred_dentist,
        preferred_time=data.preferred_time,
        preferred_date_from=data.preferred_date_from,
        preferred_date_to=data.preferred_date_to,
        status="Waiting",
        created_at=datetime.now()
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    position = db.query(WaitlistDB).filter(
        WaitlistDB.status == "Waiting",
        WaitlistDB.procedure_type == data.procedure_type,
        WaitlistDB.id <= entry.id
    ).count()

    return {
        "id": entry.id,
        "message": f"Added to waitlist. You are #{position} in queue for {data.procedure_type}.",
        "position": position,
        "estimatedWait": f"{position * 2}-{position * 5} days"
    }

@app.get("/waitlist")
async def get_waitlist(status: Optional[str] = None, db: Session = Depends(get_db)):
    """Get all waitlist entries"""
    q = db.query(WaitlistDB)
    if status:
        q = q.filter(WaitlistDB.status == status)
    entries = q.order_by(WaitlistDB.created_at).all()
    return [{
        "id": e.id, "patientName": e.patient_name, "procedureType": e.procedure_type,
        "preferredDentist": e.preferred_dentist, "preferredTime": e.preferred_time,
        "status": e.status, "createdAt": e.created_at.isoformat() if e.created_at else None,
        "notifiedAt": e.notified_at.isoformat() if e.notified_at else None,
    } for e in entries]

@app.delete("/waitlist/{entry_id}")
async def leave_waitlist(entry_id: int, db: Session = Depends(get_db)):
    """Remove from waitlist"""
    entry = db.query(WaitlistDB).filter(WaitlistDB.id == entry_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="Waitlist entry not found")
    db.delete(entry)
    db.commit()
    return {"message": "Removed from waitlist"}


# ============================================================================
# NOTIFICATION LOG ENDPOINTS
# ============================================================================

@app.get("/notifications")
async def get_notifications(appointment_id: Optional[int] = None, limit: int = 50, db: Session = Depends(get_db)):
    """Get notification history"""
    q = db.query(NotificationLogDB)
    if appointment_id:
        q = q.filter(NotificationLogDB.appointment_id == appointment_id)
    logs = q.order_by(NotificationLogDB.sent_at.desc()).limit(limit).all()
    return [{
        "id": l.id, "appointmentId": l.appointment_id, "patientName": l.patient_name,
        "channel": l.channel, "type": l.notification_type, "message": l.message,
        "status": l.status, "sentAt": l.sent_at.isoformat() if l.sent_at else None,
    } for l in logs]


# ============================================================================
# ENHANCED ANALYTICS
# ============================================================================

@app.get("/analytics/no-show-trends")
async def get_no_show_trends(db: Session = Depends(get_db)):
    """No-show rate by procedure type and risk category"""
    all_apts = db.query(AppointmentDB).all()
    if not all_apts:
        return {"message": "No data yet", "trends": {}}

    by_procedure = {}
    by_risk = {"Low": {"total": 0, "noshow": 0}, "Medium": {"total": 0, "noshow": 0}, "High": {"total": 0, "noshow": 0}}

    for a in all_apts:
        proc = a.procedure_type or "Unknown"
        if proc not in by_procedure:
            by_procedure[proc] = {"total": 0, "noshow": 0, "confirmed": 0}
        by_procedure[proc]["total"] += 1
        if a.status == "No-Show":
            by_procedure[proc]["noshow"] += 1
        if a.status == "Confirmed":
            by_procedure[proc]["confirmed"] += 1

        cat = a.risk_category or "Low"
        if cat in by_risk:
            by_risk[cat]["total"] += 1
            if a.status == "No-Show":
                by_risk[cat]["noshow"] += 1

    # Calculate rates
    for proc in by_procedure:
        t = by_procedure[proc]["total"]
        by_procedure[proc]["noShowRate"] = round(by_procedure[proc]["noshow"] / t * 100, 1) if t > 0 else 0
        by_procedure[proc]["confirmRate"] = round(by_procedure[proc]["confirmed"] / t * 100, 1) if t > 0 else 0

    for cat in by_risk:
        t = by_risk[cat]["total"]
        by_risk[cat]["noShowRate"] = round(by_risk[cat]["noshow"] / t * 100, 1) if t > 0 else 0

    return {
        "byProcedure": by_procedure,
        "byRiskCategory": by_risk,
        "totalAppointments": len(all_apts),
        "overallNoShowRate": round(sum(1 for a in all_apts if a.status == "No-Show") / len(all_apts) * 100, 1)
    }


@app.get("/analytics/waitlist-stats")
async def get_waitlist_stats(db: Session = Depends(get_db)):
    """Waitlist analytics"""
    entries = db.query(WaitlistDB).all()
    return {
        "total": len(entries),
        "waiting": sum(1 for e in entries if e.status == "Waiting"),
        "notified": sum(1 for e in entries if e.status == "Notified"),
        "booked": sum(1 for e in entries if e.status == "Booked"),
        "byProcedure": {p: sum(1 for e in entries if e.procedure_type == p) for p in set(e.procedure_type for e in entries)},
    }


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
