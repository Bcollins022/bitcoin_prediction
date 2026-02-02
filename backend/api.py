from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import pandas as pd
import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.inference import InferenceEngine
from backend.data_loader import fetch_data

app = FastAPI(title="Bitcoin Price Prediction API", version="1.0.0")

# Global instance
engine: Optional[InferenceEngine] = None

@app.on_event("startup")
async def startup_event():
    global engine
    try:
        # Check if model files exist
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models', 'LSTM.pth')
        if os.path.exists(model_path):
            engine = InferenceEngine()
            print("Inference Engine loaded successfully.")
        else:
            print("Warning: Model files not found. Prediction endpoints will fail.")
    except Exception as e:
        print(f"Error loading Inference Engine: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "bitcoin-price-prediction"}

@app.get("/predict/next-12h")
def predict_next_12h():
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train models first.")
    
    try:
        df = fetch_data()
        if df.empty:
             raise HTTPException(status_code=500, detail="Failed to fetch market data.")
             
        predictions = engine.predict_next_12h(df)
        
        # Add metadata
        result = {
            "predictions": predictions,
            "last_date": df.index[-1].strftime("%Y-%m-%d %H:%M"),
            "last_close": float(df['Close'].iloc[-1])
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/future")
def predict_future(days: int = Query(1, description="Ignored, always returns 12h forecast")):
    if not engine:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train models first.")
        
    try:
        df = fetch_data()
        if df.empty:
             raise HTTPException(status_code=500, detail="Failed to fetch market data.")
             
        # Returns list of dicts: [{"date": ..., "predicted_price": ...}]
        future_preds = engine.predict_future(df, days=days)
            
        return {"forecast": future_preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
