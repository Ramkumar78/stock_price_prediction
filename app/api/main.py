from fastapi import FastAPI, HTTPException, Request, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import json
import uuid
from typing import List, Dict, Any, Optional
import logging
import glob

# Import core modules
from app.core.download_data import download_asset_data, save_data, TICKERS, START_DATE
from app.core.feature_engineering import create_all_features, load_data_for_features, create_target_variable
from app.core.custom_pipeline import run_custom_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SPY Stock Prediction API", version="0.2.0")

# --- Security & CORS ---
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
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

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Added connect-src to allow frontend to talk to backend
        # Added script-src and style-src to support React/Vite development
        response.headers["Content-Security-Policy"] = "default-src 'self'; connect-src 'self' http://localhost:8000; img-src 'self' data: https:; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# --- Pydantic Models ---
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

class CustomTrainRequest(BaseModel):
    ticker: str
    model: str

# In-memory job store
custom_jobs = {}

# --- Endpoints ---

@app.post("/custom/train")
async def start_custom_train(request: CustomTrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    custom_jobs[job_id] = {
        "status": "pending",
        "ticker": request.ticker,
        "model": request.model,
        "progress": "Queued"
    }

    background_tasks.add_task(run_custom_pipeline, job_id, request.ticker, request.model, custom_jobs)

    return {"job_id": job_id, "status": "started"}

@app.get("/custom/status/{job_id}")
async def get_custom_status(job_id: str):
    if job_id not in custom_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return custom_jobs[job_id]

@app.post("/data/refresh", response_model=DataRefreshResponse)
async def refresh_data():
    try:
        downloaded = []
        for name, symbol in TICKERS.items():
            df = download_asset_data(symbol, name, START_DATE)
            if df is not None:
                save_data(df, name)
                downloaded.append(name)
        return {"message": "Data download complete.", "downloaded": downloaded}
    except Exception as e:
        logger.error(f"Error refreshing data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/generate", response_model=FeatureGenerationResponse)
async def generate_features():
    try:
        data = load_data_for_features(data_dir='data')
        if data['spy'] is None:
             raise HTTPException(status_code=400, detail="SPY data not found. Please refresh data first.")

        features = create_all_features(
            spy=data['spy'], vix=data['vix'], tlt=data['tlt'], dxy=data['dxy'], gld=data['gld'],
            include_regime_dependent=True
        )
        y = create_target_variable(data['spy'], forward_days=3)
        final_df = pd.concat([features, y.rename('target')], axis=1)

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(os.path.join(output_dir, 'spy_features_full.csv'))

        return {
            "message": "Features generated successfully.",
            "features_count": len(features.columns),
            "data_shape": final_df.shape
        }
    except Exception as e:
        logger.error(f"Error generating features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{model_type}")
async def train_model(model_type: str):
    import subprocess, sys

    # Map friendly names to scripts
    script_map = {
        "xgboost": "app/models/modelling_xgboost.py",
        "lightgbm": "app/models/modelling_lightgbm.py",
        "catboost": "app/models/modelling_catboost.py",
        "ensemble": "app/models/modelling_ensemble.py"
    }

    if model_type not in script_map:
        raise HTTPException(status_code=400, detail=f"Invalid model. Options: {list(script_map.keys())}")

    try:
        # Run the training script
        logger.info(f"Starting training for {model_type}...")
        result = subprocess.run([sys.executable, script_map[model_type]], capture_output=True, text=True, cwd=os.getcwd())

        if result.returncode != 0:
             logger.error(f"Training failed: {result.stderr}")
             raise HTTPException(status_code=500, detail=f"Training script failed: {result.stderr}")

        return {"message": f"{model_type.capitalize()} model trained successfully.", "output": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict", response_model=PredictResponse)
async def predict(model: str = Query("lightgbm", enum=["lightgbm", "xgboost", "catboost", "ensemble"])):
    """
    Predicts using the specified model. Defaults to LightGBM.
    """
    import lightgbm as lgb
    import xgboost as xgb
    import joblib

    # Define paths based on model selection
    base_path = f'output/models/{model}'

    # Handle filename variations based on your repo structure
    if model == 'ensemble':
        model_file = os.path.join(base_path, 'ensemble_model.pkl')
    elif model == 'catboost':
        # CatBoost specific handling if needed, usually .json or .cbm
        model_file = os.path.join(base_path, f'{model}_model_selected_features.json')
    else:
        model_file = os.path.join(base_path, f'{model}_model_selected_features.json')

    features_file = os.path.join(base_path, 'selected_features.txt')
    data_path = 'output/spy_features_full.csv'

    if not os.path.exists(model_file):
         raise HTTPException(status_code=404, detail=f"Model file not found for {model}. Please train it first.")

    try:
        # Load Data
        df = pd.read_csv(data_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        # 1. Load Features List (if applicable)
        required_features = []
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                required_features = [line.strip() for line in f.readlines()]
        else:
            # Fallback for ensemble or full models if txt doesn't exist
            required_features = df.columns.tolist()
            if 'target' in required_features: required_features.remove('target')

        # 2. Prepare Input Data
        # Ensure we have the required features in the dataframe
        available_features = [f for f in required_features if f in df.columns]
        if len(available_features) != len(required_features):
            missing = set(required_features) - set(df.columns)
            logger.warning(f"Missing features in data for prediction: {missing}")
            # Depending on model, this might crash or work.
            # We proceed with what we have if the model allows, but usually we should fail or fill.
            # For simplicity, we just select what is available and hope order is handled by model or names.
            # LightGBM/XGBoost by JSON usually rely on feature names if loaded that way, or order.
            # If using sklearn wrapper interface, they often require strict order/columns.

        latest_data = df[required_features].iloc[-1:]

        # 3. Predict based on Model Type
        prob_up = 0.0

        if model == 'lightgbm':
            booster = lgb.Booster(model_file=model_file)
            prob_up = booster.predict(latest_data)[0]

        elif model == 'xgboost':
            reg = xgb.XGBClassifier()
            reg.load_model(model_file)
            prob_up = reg.predict_proba(latest_data)[0][1] # Probability of class 1

        elif model == 'ensemble':
            ensemble = joblib.load(model_file)
            # Ensemble likely expects a specific input format, ensure it matches training
            prob_up = ensemble.predict_proba(latest_data)[0][1]

        elif model == 'catboost':
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier()
            cb.load_model(model_file, format='json')
            prob_up = cb.predict_proba(latest_data)[1]

        prediction = "UP" if prob_up > 0.5 else "DOWN"

        return {
            "prediction": prediction,
            "probability": float(prob_up),
            "model_used": model.capitalize()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(model: str = Query("lightgbm", enum=["lightgbm", "xgboost", "catboost", "ensemble"])):
    """
    Returns metrics for the specified model.
    """
    metrics_path = f'output/models/{model}/metrics.json'

    if not os.path.exists(metrics_path):
         raise HTTPException(status_code=404, detail=f"Metrics not found for {model}. Train it first.")

    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        return {
            "model": model.capitalize(),
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading metrics")
