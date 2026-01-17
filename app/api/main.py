from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import json
from typing import List, Dict, Any, Optional
import glob
import logging

# Import core modules
from app.core.download_data import download_asset_data, save_data, TICKERS, START_DATE
from app.core.feature_engineering import create_all_features, load_data_for_features, create_target_variable, remove_correlated_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SPY Stock Prediction API", description="API for downloading data, generating features, and training models for SPY prediction.", version="0.1.0")

# Security Hardening: CORS
origins = [
    "http://localhost:5173", # Vite default
    "http://localhost:3000", # React default
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Hardening: Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

app.add_middleware(SecurityHeadersMiddleware)

class DataRefreshResponse(BaseModel):
    message: str
    downloaded: List[str]

class FeatureGenerationResponse(BaseModel):
    message: str
    features_count: int
    data_shape: List[int]

class PredictResponse(BaseModel):
    prediction: str
    probability: float
    model_used: str

class MetricsResponse(BaseModel):
    model: str
    metrics: Dict[str, Any]

@app.post("/data/refresh", response_model=DataRefreshResponse)
async def refresh_data():
    """
    Downloads the latest OHLCV data for SPY, VIX, TLT, DXY, and GLD.
    """
    downloaded = []
    try:
        for name, symbol in TICKERS.items():
            df = download_asset_data(symbol, name, START_DATE)
            if df is not None:
                save_data(df, name)
                downloaded.append(name)

        return {"message": "Data download complete.", "downloaded": downloaded}
    except Exception as e:
        logger.error(f"Error refreshing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/features/generate", response_model=FeatureGenerationResponse)
async def generate_features():
    """
    Generates features from the downloaded data.
    """
    try:
        data = load_data_for_features(data_dir='data')

        if data['spy'] is None:
             raise HTTPException(status_code=400, detail="SPY data not found. Please refresh data first.")

        # Generate Features
        features = create_all_features(
            spy=data['spy'],
            vix=data['vix'],
            tlt=data['tlt'],
            dxy=data['dxy'],
            gld=data['gld'],
            include_regime_dependent=True
        )

        # Create target variable
        y = create_target_variable(data['spy'], forward_days=3)

        # Combine
        final_df = pd.concat([features, y.rename('target')], axis=1)

        # Save
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, 'spy_features_full.csv')
        final_df.to_csv(output_path)

        return {
            "message": "Features generated successfully.",
            "features_count": len(features.columns),
            "data_shape": final_df.shape
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/train/{model_type}")
async def train_model(model_type: str):
    """
    Triggers training for the specified model type.
    """
    import subprocess
    import sys

    script_map = {
        "xgboost": "app/models/modelling_xgboost.py",
        "lightgbm": "app/models/modelling_lightgbm.py",
        "catboost": "app/models/modelling_catboost.py",
        "ensemble": "app/models/modelling_ensemble.py"
    }

    if model_type not in script_map:
        raise HTTPException(status_code=400, detail=f"Invalid model type. Choose from {list(script_map.keys())}")

    script_path = script_map[model_type]

    try:
        # Using subprocess to run the script
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode != 0:
             logger.error(f"Training failed for {model_type}: {result.stderr}")
             raise HTTPException(status_code=500, detail="Training failed. Check server logs.")

        return {"message": f"{model_type} model trained successfully.", "output": result.stdout}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model {model_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/predict", response_model=PredictResponse)
async def predict():
    """
    Loads the saved LightGBM model (best performing) and returns prediction for next 3 days.
    """
    import lightgbm as lgb
    import numpy as np

    model_path_json = 'output/models/lightgbm/lightgbm_model_selected_features.json'
    features_path = 'output/models/lightgbm/selected_features.txt'
    data_path = 'output/spy_features_full.csv'

    if not os.path.exists(model_path_json) or not os.path.exists(features_path):
         raise HTTPException(status_code=404, detail="Model or features file not found. Train the model first.")

    try:
        # Load model
        model = lgb.Booster(model_file=model_path_json)

        # Load required features
        with open(features_path, 'r') as f:
            required_features = [line.strip() for line in f.readlines()]

        # Load data
        if not os.path.exists(data_path):
             raise HTTPException(status_code=404, detail="Features data not found.")

        df = pd.read_csv(data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # Get latest data
        # Ensure we have the required features
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
             logger.error(f"Missing features in data: {missing_features}")
             raise HTTPException(status_code=500, detail="Internal Server Error: Data mismatch.")

        latest = df[required_features].iloc[-1:]

        # Predict
        prob_up = model.predict(latest)[0]
        prediction = "UP" if prob_up > 0.5 else "DOWN"

        return {
            "prediction": prediction,
            "probability": float(prob_up),
            "model_used": "LightGBM (Selected Features)"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Returns the metrics for the best model.
    """
    metrics_path = 'output/models/lightgbm/metrics.json'

    if not os.path.exists(metrics_path):
         raise HTTPException(status_code=404, detail="Metrics not found.")

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        return {
            "model": "LightGBM",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
