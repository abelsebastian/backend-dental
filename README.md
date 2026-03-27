# Smart DentalOps Backend

## 🚀 Quick Start

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Train the Model (First Time Only)
```bash
# This creates the saved_model.h5 file
python model.py
```

### Start the Server
```bash
# Start FastAPI server with auto-reload
uvicorn main:app --reload
```
Backend will run on `http://localhost:8000`

## 📁 Project Structure
```
backend/
├── main.py              # FastAPI server and API endpoints
├── model.py             # ANN training script
├── requirements.txt     # Python dependencies
├── dataset.csv          # Generated training data
├── saved_model.h5       # Trained neural network
└── README.md           # This file
```

## 🔌 API Endpoints

### POST /predict
Predict no-show risk for a patient appointment

**Request Body:**
```json
{
  "name": "John Doe",
  "age": 35,
  "procedure": "cleaning",
  "previousNoShow": false
}
```

**Response:**
```json
{
  "risk": "25.5%",
  "duration": "20 min",
  "slot": "10:30 AM (Confirmed)"
}
```

### GET /
Health check endpoint

### GET /docs
Interactive API documentation (Swagger UI)

### GET /model-info
Get information about the loaded model

## 🧠 Model Details

### Architecture
- Input Layer: 5 neurons (age, procedure, previous no-show, day, time)
- Hidden Layer 1: 8 neurons (ReLU activation)
- Hidden Layer 2: 4 neurons (ReLU activation)
- Output Layer: 1 neuron (Sigmoid activation)

### Training
- Dataset: 1000 synthetic samples
- Training/Test Split: 80/20
- Epochs: 50
- Batch Size: 16
- Optimizer: Adam
- Loss Function: Binary Crossentropy

## 🔧 Configuration

### CORS Settings
Currently allows all origins for development. In production, update `main.py`:
```python
allow_origins=["http://localhost:5173"]  # Specific frontend URL
```

### Port Configuration
Default port is 8000. To change:
```bash
uvicorn main:app --reload --port 8001
```

## 🐛 Troubleshooting

### "Model file not found"
**Solution**: Run `python model.py` to train and save the model

### "Module not found"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### "Port already in use"
**Solution**: Kill the process or use a different port

### "CORS error from frontend"
**Solution**: Check that CORS middleware is properly configured in `main.py`

## 📚 Learning Resources
Read `docs/backend-guide.md` for detailed explanations of:
- FastAPI concepts
- API endpoint design
- Model integration
- Business logic implementation

## 🧪 Testing the API

### Using Browser
Visit `http://localhost:8000/docs` for interactive API testing

### Using curl
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Patient",
    "age": 30,
    "procedure": "cleaning",
    "previousNoShow": false
  }'
```

### Using Python
```python
import requests

response = requests.post('http://localhost:8000/predict', json={
    'name': 'Test Patient',
    'age': 30,
    'procedure': 'cleaning',
    'previousNoShow': False
})

print(response.json())
```

## 📝 Notes for Students

### Key Concepts to Understand
1. **FastAPI**: Modern Python web framework
2. **Pydantic**: Data validation using Python type hints
3. **CORS**: Allows frontend to communicate with backend
4. **Neural Network**: Predicts patterns from training data
5. **API Endpoints**: URLs that accept requests and return responses

### For Your Viva
Be prepared to explain:
- How the ANN makes predictions
- What each layer in the network does
- How data flows from frontend to backend
- Why we normalize input data
- What CORS is and why it's needed

## 🚀 Next Steps
After understanding this backend:
1. Add more endpoints (get all patients, appointments)
2. Implement proper error handling
3. Add logging for debugging
4. Create unit tests
5. Add database integration (SQLite/PostgreSQL)

## ⚠️ Important Notes
- This is an academic prototype, not production-ready
- Uses synthetic data for training
- No authentication/authorization
- No database (uses in-memory data)
- Simplified business logic

---
**Built for learning | Academic Project 2026**
