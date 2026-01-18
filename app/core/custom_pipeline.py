import pandas as pd
import os
import logging
import traceback
from datetime import datetime
from app.core.download_data import download_asset_data
from app.core.feature_engineering import create_all_features, create_target_variable, load_data_for_features

# Import model logic
from app.models.modelling_xgboost import (
    load_and_prepare_data,
    split_train_test,
    handle_missing_values,
    train_xgboost,
    evaluate_model as evaluate_xgboost,
    XGBOOST_PARAMS
)
from app.models.modelling_lightgbm import (
    train_lightgbm,
    evaluate_model as evaluate_lightgbm,
    LIGHTGBM_PARAMS
)

import xgboost as xgb

logger = logging.getLogger(__name__)

async def run_custom_pipeline(job_id: str, ticker: str, model_type: str, jobs_dict: dict):
    try:
        jobs_dict[job_id]["status"] = "running"
        jobs_dict[job_id]["progress"] = f"Downloading data for {ticker}..."

        # 1. Download Data
        # Download the target asset
        df = download_asset_data(ticker, ticker, '2015-01-01')
        if df is None:
            raise Exception(f"Failed to download data for {ticker}")

        # Load existing macro data if available (optional context)
        # We don't fail if they are missing
        macro_data = load_data_for_features()

        jobs_dict[job_id]["progress"] = "Generating features..."

        # 2. Feature Engineering
        # We use the downloaded df as the main 'spy' input
        features = create_all_features(
            spy=df,
            vix=macro_data.get('vix'),
            tlt=macro_data.get('tlt'),
            dxy=macro_data.get('dxy'),
            gld=macro_data.get('gld'),
            include_regime_dependent=True
        )

        y = create_target_variable(df, forward_days=3)
        final_df = pd.concat([features, y.rename('target')], axis=1)

        # Save to a temporary location for this job
        output_dir = f"output/custom_models/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        data_path = os.path.join(output_dir, "features.csv")
        final_df.to_csv(data_path)

        jobs_dict[job_id]["progress"] = f"Training {model_type}..."

        # 3. Load and Prepare
        # Reuse logic from modelling scripts
        X, y, dates = load_and_prepare_data(data_path)

        # Determine split dates
        # Use standard split if data allows, otherwise fallback?
        # Standard: Train < 2024, Test >= 2024
        train_end = '2023-12-31'
        test_start = '2024-01-01'

        # Check if we have data for the test set
        if dates.max() < pd.Timestamp(test_start):
             # Fallback: Last 20% as test
             dates_list = dates.unique()
             split_idx = int(len(dates_list) * 0.8)
             split_date = dates_list[split_idx]
             train_end = pd.Timestamp(split_date).strftime('%Y-%m-%d')
             # Next day
             test_start = (pd.Timestamp(split_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
             logger.info(f"Data ends before 2024. Using dynamic split: Train up to {train_end}")

        X_train, X_test, y_train, y_test = split_train_test(
            X, y, dates, train_end, test_start
        )

        if len(X_test) == 0:
             # Just in case
             raise Exception("No test data available after split.")

        # Handle Missing
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = handle_missing_values(
            X_train, X_test, y_train, y_test
        )

        # 4. Train & Evaluate
        metrics = {}
        model = None

        if model_type == 'xgboost':
            model, evals = train_xgboost(
                X_train_clean, y_train_clean,
                X_test_clean, y_test_clean,
                XGBOOST_PARAMS.copy()
            )
            metrics, y_pred, y_prob = evaluate_xgboost(
                model, X_train_clean, y_train_clean, X_test_clean, y_test_clean
            )
            model.save_model(os.path.join(output_dir, "model.json"))

        elif model_type == 'lightgbm':
             model, evals = train_lightgbm(
                X_train_clean, y_train_clean,
                X_test_clean, y_test_clean,
                LIGHTGBM_PARAMS.copy()
            )
             metrics, y_pred, y_prob = evaluate_lightgbm(
                model, X_train_clean, y_train_clean, X_test_clean, y_test_clean
            )
             model.save_model(os.path.join(output_dir, "model.json"))

        else:
            raise Exception(f"Model {model_type} not supported for custom training yet.")

        # 5. Prediction for latest data
        # We need the most recent feature vector.
        # It might be in X_test, or it might have been dropped due to NaNs if it's too recent?
        # Actually handle_missing_values drops rows with NaNs.
        # We should try to get the very last row from the full X (before nan dropping)
        # but we need it to be populated.
        # If the last few days have NaNs (e.g. forward target), we can still predict for them
        # because prediction doesn't need 'target'.
        # However, 'target' is in 'final_df'.
        # The 'features' df (from create_all_features) has all features.

        # Let's get the last row from 'features' df directly
        last_row = features.iloc[[-1]]

        # Check for NaNs in features
        if last_row.isna().any().any():
             # Try to find the last fully valid row
             last_valid_idx = features.dropna().index[-1]
             last_row = features.loc[[last_valid_idx]]
             logger.warning(f"Last row had NaNs. Using last valid row from {last_valid_idx}")

        prob = 0.0
        if model_type == 'xgboost':
            dmatrix = xgb.DMatrix(last_row)
            prob = model.predict(dmatrix)[0]
        elif model_type == 'lightgbm':
            prob = model.predict(last_row)[0]

        prediction = "UP" if prob > 0.5 else "DOWN"

        jobs_dict[job_id]["status"] = "completed"
        jobs_dict[job_id]["result"] = {
            "metrics": metrics,
            "prediction": prediction,
            "probability": float(prob),
            "ticker": ticker,
            "model": model_type,
            "last_date": str(last_row.index[0])
        }
        jobs_dict[job_id]["progress"] = "Done"

    except Exception as e:
        logger.error(f"Custom pipeline failed: {e}")
        traceback.print_exc()
        jobs_dict[job_id]["status"] = "failed"
        jobs_dict[job_id]["error"] = str(e)
