import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from datetime import datetime
import logging
from sklearn.datasets import fetch_california_housing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Load and preprocess the housing data with feature engineering"""
    try:
        logger.info("[DEBUG] Fetching California housing dataset...")
        california = fetch_california_housing()
        data = pd.DataFrame(california.data, columns=california.feature_names)
        data['MedHouseVal'] = california.target
        
        logger.info("[DEBUG] Original dataset loaded with shape: %s", data.shape)
        
        # Feature engineering
        logger.info("[DEBUG] Performing feature engineering...")
        data['MedInc_squared'] = data['MedInc'] ** 2
        data['RoomsPerBedroom'] = data['AveRooms'] / data['AveBedrms']
        data['PopulationDensity'] = data['Population'] / (abs(data['Latitude'] * data['Longitude']) + 1)
        
        # Handle infinite values from division
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        
        logger.info("[DEBUG] Dataset after feature engineering shape: %s", data.shape)
        return data
    except Exception as e:
        logger.error("[ERROR] Failed to load and preprocess data: %s", str(e))
        raise

def prepare_data(data):
    """Prepare train/test sets and scale features"""
    try:
        logger.info("[DEBUG] Preparing train/test splits...")
        features = [
            'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
            'Latitude', 'Longitude', 'MedInc_squared', 'RoomsPerBedroom',
            'PopulationDensity'
        ]
        X = data[features]
        y = data['MedHouseVal']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info("[DEBUG] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        logger.info("[DEBUG] Data prepared. Train shape: %s, Test shape: %s", 
                   X_train_scaled.shape, X_test_scaled.shape)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, features
    except Exception as e:
        logger.error("[ERROR] Failed to prepare data: %s", str(e))
        raise

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model"""
    try:
        logger.info("[DEBUG] Training XGBoost model...")
        start_time = datetime.now()
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("[DEBUG] XGBoost training completed in %s", datetime.now() - start_time)
        logger.info("[RESULTS] XGBoost Performance:")
        logger.info("RMSE: %.4f", rmse)
        logger.info("MAE: %.4f", mae)
        logger.info("R²: %.4f", r2)
        
        return model, rmse, mae, r2
    except Exception as e:
        logger.error("[ERROR] Failed to train XGBoost model: %s", str(e))
        raise

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model"""
    try:
        logger.info("[DEBUG] Training Random Forest model...")
        start_time = datetime.now()
        
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info("[DEBUG] Random Forest training completed in %s", datetime.now() - start_time)
        logger.info("[RESULTS] Random Forest Performance:")
        logger.info("RMSE: %.4f", rmse)
        logger.info("MAE: %.4f", mae)
        logger.info("R²: %.4f", r2)
        
        plot_feature_importance(model, X_train)
        
        return model, rmse, mae, r2
    except Exception as e:
        logger.error("[ERROR] Failed to train Random Forest model: %s", str(e))
        raise

def plot_feature_importance(model, X_train):
    """Plot and save feature importance for Random Forest"""
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance - Random Forest")
        plt.bar(range(X_train.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices, rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        plt.tight_layout()
        
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/rf_feature_importance.png')
        plt.close()
        logger.info("[DEBUG] Feature importance plot saved")
    except Exception as e:
        logger.error("[ERROR] Failed to plot feature importance: %s", str(e))

def save_models(xgb_model, rf_model, scaler, features):
    """Save trained models and scaler to disk"""
    try:
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(xgb_model, 'models/house_model_xgb.pkl')
        joblib.dump(rf_model, 'models/house_model_rf.pkl')
        joblib.dump(scaler, 'models/house_scaler.pkl')
        joblib.dump(features, 'models/house_features.pkl')
        
        logger.info("[DEBUG] Models and scaler saved successfully")
    except Exception as e:
        logger.error("[ERROR] Failed to save models: %s", str(e))
        raise

def main():
    try:
        logger.info("=== Starting House Price Model Training ===")
        
        data = load_and_preprocess_data()
        X_train, X_test, y_train, y_test, scaler, features = prepare_data(data)
        
        xgb_model, xgb_rmse, xgb_mae, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test)
        rf_model, rf_rmse, rf_mae, rf_r2 = train_random_forest(X_train, y_train, X_test, y_test)
        
        save_models(xgb_model, rf_model, scaler, features)
        
        logger.info("=== Model Training Completed Successfully ===")
        logger.info("Final Model Performance:")
        logger.info("XGBoost - RMSE: %.4f, MAE: %.4f, R²: %.4f", xgb_rmse, xgb_mae, xgb_r2)
        logger.info("Random Forest - RMSE: %.4f, MAE: %.4f, R²: %.4f", rf_rmse, rf_mae, rf_r2)
        
    except Exception as e:
        logger.error("!!! Model Training Failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()
