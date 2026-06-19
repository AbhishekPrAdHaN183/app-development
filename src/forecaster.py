import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "forecaster_pipeline.joblib")

def load_data_from_transactions(transactions: list) -> pd.DataFrame:
    """Convert list of SalesTransaction objects into a daily aggregated DataFrame."""
    if not transactions:
        return pd.DataFrame(columns=["date", "units_sold", "unit_price", "is_promo"])
    
    # Extract records into dict list
    data = []
    for tx in transactions:
        data.append({
            "date": tx.date,
            "units_sold": tx.units_sold,
            "unit_price": tx.unit_price,
            "is_promo": 1 if tx.is_promo else 0
        })
    
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    
    # Aggregate to daily levels
    daily = df.groupby("date").agg(
        units_sold=pd.NamedAgg(column="units_sold", aggfunc="sum"),
        unit_price=pd.NamedAgg(column="unit_price", aggfunc="mean"),
        is_promo=pd.NamedAgg(column="is_promo", aggfunc="max")  # 1 if any product was promo on that day
    ).reset_index()
    
    # Sort chronologically
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily

def engineer_features(df: pd.DataFrame, lag_days: int = 7, rolling_window: int = 7) -> pd.DataFrame:
    """
    Perform feature engineering: calendar features, lag features, and rolling metrics.
    Ensures that features do not introduce future data leakage.
    """
    # Create a copy to avoid side effects
    feat_df = df.copy()
    
    # Calendar/Temporal Features
    feat_df["day_of_week"] = feat_df["date"].dt.dayofweek
    feat_df["month"] = feat_df["date"].dt.month
    feat_df["day_of_year"] = feat_df["date"].dt.dayofyear
    feat_df["is_weekend"] = feat_df["day_of_week"].isin([5, 6]).astype(int)
    
    # Generate lags for target: units_sold
    for lag in range(1, lag_days + 1):
        feat_df[f"sales_lag_{lag}"] = feat_df["units_sold"].shift(lag)
    
    # Extra key lags for weekly cycles
    feat_df["sales_lag_14"] = feat_df["units_sold"].shift(14)
    
    # Rolling averages and standard deviations (shifted by 1 to prevent data leakage)
    # The rolling mean of units_sold for today should only look at yesterday and prior!
    feat_df["rolling_mean_7"] = feat_df["units_sold"].shift(1).rolling(window=7, min_periods=1).mean()
    feat_df["rolling_mean_30"] = feat_df["units_sold"].shift(1).rolling(window=30, min_periods=1).mean()
    feat_df["rolling_std_7"] = feat_df["units_sold"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)
    
    # Drop rows containing NaN due to lag offsets
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df

def train_forecaster(df: pd.DataFrame, config: dict) -> dict:
    """
    Train a forecasting regression model using daily sales data.
    Uses temporal splitting to evaluate performance.
    """
    # 1. Engineer features
    lag_days = config.get("lag_days", 7)
    rolling_window = config.get("rolling_window", 7)
    algorithm = config.get("algorithm", "random_forest")
    test_size = config.get("test_size", 0.2)
    
    feat_df = engineer_features(df, lag_days=lag_days, rolling_window=rolling_window)
    if len(feat_df) < 30:
        raise ValueError(f"Insufficient daily data points ({len(feat_df)}) after feature engineering. Seed more data.")
    
    # Define features and target
    # Exclude non-feature columns
    exclude_cols = ["date", "units_sold"]
    feature_cols = [col for col in feat_df.columns if col not in exclude_cols]
    
    X = feat_df[feature_cols]
    y = feat_df["units_sold"]
    
    # 2. Temporal Train-Test Split (strictly chronological)
    split_idx = int(len(feat_df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = feat_df["date"].iloc[split_idx:]
    
    # 3. Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Instantiate and fit model
    if algorithm == "linear_regression":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate on Test set
    y_pred = model.predict(X_test_scaled)
    # Ensure no negative sales are predicted
    y_pred = np.clip(y_pred, 0, None)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Train final model on all data to maximize data utilization for future forecasts
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    
    if algorithm == "linear_regression":
        model_final = Ridge(alpha=1.0)
    else:
        model_final = RandomForestRegressor(n_estimators=100, random_state=42)
        
    model_final.fit(X_scaled, y)
    
    # 6. Save model pipeline
    pipeline = {
        "model": model_final,
        "scaler": scaler_final,
        "feature_cols": feature_cols,
        "lag_days": lag_days,
        "rolling_window": rolling_window,
        "algorithm": algorithm,
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    }
    joblib.dump(pipeline, MODEL_FILE)
    
    # Return metrics and split predictions for evaluation charts
    test_eval = []
    for dt, actual, pred in zip(dates_test, y_test, y_pred):
        test_eval.append({
            "date": dt.strftime("%Y-%m-%d"),
            "actual": float(actual),
            "predicted": float(pred)
        })
        
    return {
        "metrics": pipeline["metrics"],
        "algorithm": algorithm,
        "test_eval": test_eval,
        "feature_importance": get_feature_importance(model_final, feature_cols)
    }

def get_feature_importance(model, feature_cols: list) -> list[dict]:
    """Extract coefficients or feature importances from the estimator."""
    importances = []
    if hasattr(model, "feature_importances_"):
        # Tree-based
        for col, val in zip(feature_cols, model.feature_importances_):
            importances.append({"feature": col, "importance": float(val)})
    elif hasattr(model, "coef_"):
        # Linear
        coefs = np.abs(model.coef_)
        total = np.sum(coefs) or 1.0
        for col, val in zip(feature_cols, model.coef_):
            importances.append({"feature": col, "importance": float(np.abs(val) / total)})
            
    importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
    return importances[:6]  # Return top 6 features

def forecast_future_demand(df: pd.DataFrame, steps: int = 30) -> list[dict]:
    """
    Perform multi-step recursive forecasting for the next N days.
    Simulates lag updates day-by-day.
    """
    if not os.path.exists(MODEL_FILE):
        return []
    
    # Load model pipeline
    pipeline = joblib.load(MODEL_FILE)
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    feature_cols = pipeline["feature_cols"]
    lag_days = pipeline["lag_days"]
    
    # Verify we have enough recent data
    if len(df) < 30:
        return []
    
    forecast_df = df.copy()
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])
    
    # Get last known values
    last_date = forecast_df["date"].max()
    last_price = forecast_df["unit_price"].iloc[-1]
    
    predictions = []
    
    for i in range(1, steps + 1):
        next_date = last_date + timedelta(days=i)
        
        # Build temp dataframe containing future row to engineer lags
        temp_row = pd.DataFrame([{
            "date": next_date,
            "units_sold": 0.0,  # Placeholder, will be computed
            "unit_price": last_price,  # Assume pricing remains constant
            "is_promo": 0  # Assume no promo in the forecast horizon unless specified
        }])
        
        # Append to forecast dataframe to run standard feature engineering
        temp_forecast_df = pd.concat([forecast_df, temp_row], ignore_index=True)
        
        # Engineer features
        temp_feat = engineer_features(temp_forecast_df, lag_days=lag_days)
        
        # Extract features for the new row (last row)
        X_row = temp_feat[feature_cols].iloc[[-1]]
        X_scaled = scaler.transform(X_row)
        
        # Predict units sold
        pred_units = model.predict(X_scaled)[0]
        pred_units = max(0, float(pred_units))  # Ensure non-negative
        
        # Store prediction
        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_demand": round(pred_units, 1)
        })
        
        # Update units_sold in forecast_df for subsequent steps (recursive step)
        temp_row["units_sold"] = pred_units
        forecast_df = pd.concat([forecast_df, temp_row], ignore_index=True)
        
    return predictions
