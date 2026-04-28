from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Request format
class SymptomInput(BaseModel):
    text: str

# ✅ API endpoint
@app.post("/predict")
def predict(data: SymptomInput):
    input_data = vectorizer.transform([data.text])
    prediction = model.predict(input_data)[0]

    return {
        "disease": prediction,
        "advice": "This is not a medical diagnosis. Consult a doctor if needed."
    }

# ✅ Optional homepage (fixes 404 issue)
@app.get("/")
def home():
    return {"message": "AI Symptom Checker API is running!"}