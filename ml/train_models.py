"""
ML Model Training Pipeline for GA Predictive Maintenance System.

This module implements the complete training pipeline for both Random Forest and
LSTM models to predict Remaining Useful Life (RUL) of aircraft components.

The pipeline includes:
1. Data extraction from SQLite database
2. Data validation and cleaning
3. Feature engineering (pivoting, normalization, sequence generation)
4. Model training (Random Forest + LSTM)
5. Model evaluation and metrics logging
6. Model persistence and metadata storage
7. Prediction logging to database

This is a core component of the Model Integration Pipeline.

Author: Ndubuisi Chibuogwu
Date: Dec 2024 - July 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import joblib
import json

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# TensorFlow/Keras imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Project settings and paths
from config.settings import (
    DATABASE_PATH,
    TOP_SENSOR_PARAMS,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    LSTM_SEQUENCE_LENGTH,
    LSTM_UNITS,
    LSTM_LEARNING_RATE,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    TRAIN_TEST_SPLIT,
    RANDOM_SEED,
    MODEL_DIR,
    RF_MODEL_PATH,
    LSTM_MODEL_PATH
)
# Database helpers and logging
from config.db_utils import execute_query, execute_many, execute_write
from config.logging_config import setup_logger

# Create logger for this module
logger = setup_logger(__name__)


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def extract_sensor_data() -> pd.DataFrame:
    """
    Pull all sensor data from the database.
    """
    logger.info("Extracting sensor data from database...")

    query = "SELECT * FROM sensor_data"
    df = execute_query(query)

    logger.info(f"Extracted {len(df):,} sensor records")
    return df


def extract_components_rul() -> pd.DataFrame:
    """
    Pull component IDs and their Remaining Useful Life (RUL) values.
    """
    logger.info("Extracting component RUL data...")

    query = "SELECT component_id, remaining_useful_life FROM components"
    df = execute_query(query)

    logger.info(f"Extracted RUL for {len(df)} components")
    return df


# ============================================================================
# DATA VALIDATION AND CLEANING
# ============================================================================

def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the raw sensor data:
    - parse timestamps
    - keep only the sensors we care about
    - remove duplicate readings per (component, time, parameter)
    """
    logger.info("Cleaning sensor data...")

    initial_count = len(df)

    # Convert timestamp column to real datetime, drop rows that fail conversion
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Keep only the main sensor channels used in the models
    df = df[df['parameter'].isin(TOP_SENSOR_PARAMS)]

    # Remove any repeated rows for the same component / time / sensor
    df = df.drop_duplicates(subset=['component_id', 'timestamp', 'parameter'])

    final_count = len(df)
    logger.info(
        f"Cleaned sensor data: {initial_count:,} → {final_count:,} records "
        f"({initial_count - final_count:,} removed)"
    )

    return df


def pivot_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the data from long format into wide format.

    After pivoting, each row = one (component_id, timestamp)
    and each column = a sensor parameter (oil_press, rpm, etc.).
    """
    logger.info("Pivoting sensor data to wide format...")

    pivoted = df.pivot_table(
        index=['component_id', 'timestamp'],
        columns='parameter',
        values='value'
    ).sort_index().reset_index()

    # Fill missing values by carrying last known value forward, then backward
    pivoted.ffill(inplace=True)
    pivoted.bfill(inplace=True)

    # For safety, drop rows where any of the main sensors are still NaN
    pivoted = pivoted.dropna(subset=TOP_SENSOR_PARAMS)

    logger.info(f"Pivoted data shape: {pivoted.shape}")
    return pivoted


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Scale the sensor columns to the [0, 1] range.

    This keeps all sensors on a similar scale so the LSTM doesn't get dominated
    by one feature with larger numeric values.
    """
    logger.info("Normalizing sensor features...")

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    logger.info("Feature normalization complete")
    return df, scaler


def create_sequences(
    pivoted_df: pd.DataFrame,
    components_df: pd.DataFrame,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows for LSTM training.

    For each component, we take 'seq_length' consecutive rows and treat that
    as one sequence. The target for that sequence is the component's RUL.
    """
    logger.info(f"Creating sequences with length {seq_length}...")

    # Use max RUL to normalize targets into [0, 1]
    y_max = components_df['remaining_useful_life'].dropna().max()
    if pd.isna(y_max) or y_max == 0:
        y_max = 1.0
        logger.warning("Invalid max RUL, using default value of 1.0")

    x_seq = []
    y_seq = []
    comp_ids = []

    # Go component by component
    for comp_id in pivoted_df['component_id'].unique():
        comp_data = pivoted_df[pivoted_df['component_id'] == comp_id].sort_values('timestamp')
        rul_val = components_df.loc[
            components_df['component_id'] == comp_id, 'remaining_useful_life'
        ].values

        # Skip components with no RUL or not enough history
        if (rul_val.size == 0 or
                len(comp_data) < seq_length or
                pd.isna(rul_val[0])):
            continue

        # Slide a window of length seq_length across this component's time series
        for i in range(len(comp_data) - seq_length + 1):
            sequence = comp_data[TOP_SENSOR_PARAMS].iloc[i:i + seq_length].values
            x_seq.append(sequence)
            y_seq.append(rul_val[0])  # same RUL label for all windows of that component
            comp_ids.append(comp_id)

    # Convert to numpy arrays for model training
    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq) / y_max  # normalize RUL
    comp_ids = np.array(comp_ids)

    # Drop any sequences that still contain NaNs
    valid_mask = ~np.isnan(x_seq).any(axis=(1, 2)) & ~np.isnan(y_seq)
    x_seq = x_seq[valid_mask]
    y_seq = y_seq[valid_mask]
    comp_ids = comp_ids[valid_mask]

    logger.info(f"Created {len(x_seq):,} sequences from {len(pivoted_df['component_id'].unique())} components")

    return x_seq, y_seq, comp_ids


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_max: float
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """
    Train a Random Forest baseline.

    For Random Forest we don't feed the full time window, we only take the
    last timestep of each sequence as the feature vector.
    """
    logger.info("Training Random Forest model...")

    # Take only the last time step from each sequence
    x_rf_train = x_train[:, -1, :]
    x_rf_test = x_test[:, -1, :]

    # Drop any rows where the target is NaN
    nan_mask = ~np.isnan(y_train)
    x_rf_train = x_rf_train[nan_mask]
    y_train_clean = y_train[nan_mask]

    # Set up and train the model
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1  # use all cores
    )
    rf.fit(x_rf_train, y_train_clean)

    # Predict and bring RUL back to original scale
    y_pred = rf.predict(x_rf_test) * y_max
    y_true = y_test * y_max

    # Calculate metrics in real RUL hours
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }

    logger.info(f"Random Forest - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Store the trained model to disk
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, RF_MODEL_PATH)
    logger.info(f"Random Forest model saved to {RF_MODEL_PATH}")

    return rf, metrics


def train_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_max: float
) -> Tuple[Sequential, Dict[str, float]]:
    """
    Train the LSTM model on full sequences.
    """
    logger.info("Training LSTM model...")

    # Simple one-layer LSTM followed by a dense output
    model = Sequential([
        Input(shape=(LSTM_SEQUENCE_LENGTH, len(TOP_SENSOR_PARAMS))),
        LSTM(LSTM_UNITS),
        Dense(1)
    ])

    model.compile(optimizer=Adam(LSTM_LEARNING_RATE), loss='mse')

    logger.info(
        f"LSTM architecture: {LSTM_UNITS} units, {LSTM_EPOCHS} epochs, "
        f"batch size {LSTM_BATCH_SIZE}"
    )

    # Train and keep track of train/validation losses
    history = model.fit(
        x_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Predict and denormalize back to hours
    y_pred = model.predict(x_test).flatten() * y_max
    y_true = y_test * y_max

    # Metrics in real units
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1])
    }

    logger.info(f"LSTM - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    # Save Keras model
    model.save(str(LSTM_MODEL_PATH))
    logger.info(f"LSTM model saved to {LSTM_MODEL_PATH}")

    return model, metrics


# ============================================================================
# PREDICTION LOGGING
# ============================================================================

def log_predictions_to_database(
    component_ids: np.ndarray,
    predictions: np.ndarray,
    model_id: int = 99,
    confidence: float = 0.85,
    time_horizon: str = "100h"
) -> None:
    """
    Write model predictions into the component_predictions table.

    This makes it easy for the dashboard to show the latest RUL for each component.
    """
    logger.info(f"Logging {len(predictions)} predictions to database...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clear any previous predictions for this model id
    delete_query = "DELETE FROM component_predictions WHERE model_id = ?"
    execute_write(delete_query, (model_id,))

    # Build the rows we want to insert
    records = [
        (
            int(cid),
            model_id,
            'remaining_life',
            float(pred),
            confidence,
            time_horizon,
            'Predicted via LSTM',
            timestamp
        )
        for cid, pred in zip(component_ids, predictions)
    ]

    # Bulk insert into the DB
    insert_query = """
        INSERT INTO component_predictions (
            component_id, model_id, prediction_type, predicted_value,
            confidence, time_horizon, explanation, prediction_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows_inserted = execute_many(insert_query, records)
    logger.info(f"Successfully logged {rows_inserted} predictions")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_predictions(
    y_true: np.ndarray,
    y_rf: np.ndarray,
    y_lstm: np.ndarray,
    y_max: float
) -> None:
    """
    Plot actual vs predicted RUL for both models on one chart.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(y_true, y_rf, alpha=0.6, label="RF predictions", s=20)
    ax.scatter(y_true, y_lstm, alpha=0.6, label="LSTM predictions", s=20)
    ax.plot([0, y_max], [0, y_max], 'k--', label="Ideal", linewidth=2)

    ax.set_xlabel("Actual RUL (hours)")
    ax.set_ylabel("Predicted RUL (hours)")
    ax.set_title("Predicted vs Actual RUL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "predictions_comparison.png", dpi=150)
    logger.info("Saved predictions comparison plot")
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Run the full training pipeline end to end.
    """
    logger.info("=" * 80)
    logger.info("Starting ML Model Training Pipeline")
    logger.info("=" * 80)

    # 1) Pull data from the database
    sensor_df = extract_sensor_data()
    components_df = extract_components_rul()

    # 2) Clean and reshape it
    sensor_df = clean_sensor_data(sensor_df)
    pivoted_df = pivot_sensor_data(sensor_df)
    pivoted_df, scaler = normalize_features(pivoted_df, TOP_SENSOR_PARAMS)

    # 3) Turn time series into fixed-length sequences
    x_seq, y_seq, comp_ids = create_sequences(
        pivoted_df, components_df, LSTM_SEQUENCE_LENGTH
    )

    if len(x_seq) == 0:
        logger.error("No valid sequences generated. Cannot train models.")
        return

    # 4) Train/test split at the sequence level
    x_train, x_test, y_train, y_test, comp_train, comp_test = train_test_split(
        x_seq, y_seq, comp_ids,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED
    )

    logger.info(f"Training set: {len(x_train):,} sequences")
    logger.info(f"Test set: {len(x_test):,} sequences")

    # Use max RUL to undo normalization later
    y_max = components_df['remaining_useful_life'].dropna().max()

    # 5) Train both models
    rf_model, rf_metrics = train_random_forest(x_train, y_train, x_test, y_test, y_max)
    lstm_model, lstm_metrics = train_lstm(x_train, y_train, x_test, y_test, y_max)

    # 6) Get predictions on the test set
    y_rf_pred = rf_model.predict(x_test[:, -1, :]) * y_max
    y_lstm_pred = lstm_model.predict(x_test).flatten() * y_max

    # 7) Save LSTM predictions back into the DB for the dashboard to consume
    log_predictions_to_database(comp_test, y_lstm_pred)

    # 8) Visualize side-by-side performance of RF vs LSTM
    plot_predictions(y_test * y_max, y_rf_pred, y_lstm_pred, y_max)

    logger.info("=" * 80)
    logger.info("ML Model Training Pipeline Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
