"""
Configuration settings for the GA Predictive Maintenance System.

This module centralizes all configuration parameters, paths, and constants
used throughout the PdM system. It eliminates hard-coded values and provides
a single source of truth for system-wide settings.

Author: Ndubuisi Chibuogwu
Date: Dec 2024 - July 2025
"""

import os
from pathlib import Path
from typing import List, Dict

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent

# Database configuration
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "ga_maintenance.db"
SEED_SQL_PATH = PROJECT_ROOT / "full_pdm_seed.sql"

# Model storage
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
RF_MODEL_PATH = MODEL_DIR / "rf_model_rul.pkl"
LSTM_MODEL_PATH = MODEL_DIR / "model_lstm_rul.keras"

# Data and logs
BACKUP_DIR = PROJECT_ROOT / "backups"
LOG_DIR = PROJECT_ROOT / "logs"

# Assets
LOGO_PATH = PROJECT_ROOT / "logo.png"

# Ensure critical directories exist
for directory in [DATABASE_DIR, MODEL_DIR, BACKUP_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SENSOR PARAMETERS
# ============================================================================

# Top sensor parameters monitored across all aircraft components
TOP_SENSOR_PARAMS: List[str] = [
    'cht',                  # Cylinder Head Temperature
    'fuel_flow',           # Fuel Flow Rate
    'rpm',                  # Engine RPM
    'manifold_press',      # Manifold Pressure
    'bus_voltage',         # Electrical Bus Voltage
    'alternator_current',  # Alternator Current
    'hyd_press',           # Hydraulic Pressure
    'brake_press',         # Brake Pressure
    'oil_press',           # Oil Pressure
    'oil_temp'             # Oil Temperature
]

# Sensor units mapping
SENSOR_UNITS: Dict[str, str] = {
    'oil_press': 'psi',
    'hyd_press': 'psi',
    'brake_press': 'psi',
    'manifold_press': 'psi',
    'cht': '°C',
    'oil_temp': '°C',
    'rpm': 'rpm',
    'bus_voltage': 'volts',
    'alternator_current': 'amps',
    'fuel_flow': 'gph'
}

# Sensor health thresholds (values below threshold indicate unhealthy sensor)
SENSOR_HEALTH_THRESHOLDS: Dict[str, float] = {
    'oil_press': 30.0,
    'hyd_press': 40.0,
    'brake_press': 50.0,
    'manifold_press': 25.0,
    'cht': 100.0,
    'oil_temp': 90.0,
    'rpm': 1000.0,
    'bus_voltage': 11.0,
    'alternator_current': 15.0
}

# Sampling intervals per parameter (in seconds)
SAMPLING_INTERVALS: Dict[str, int] = {
    'oil_press': 60,
    'cht': 30,
    'rpm': 10,
    'bus_voltage': 60,
    'alternator_current': 30,
    'hyd_press': 60,
    'brake_press': 60,
    'manifold_press': 30,
    'oil_temp': 60,
    'fuel_flow': 30
}


# ============================================================================
# SYNTHETIC DATA GENERATION PARAMETERS
# ============================================================================

# Number of components to generate synthetic data for
DEFAULT_NUM_COMPONENTS: int = 10

# Number of sensor records per component
DEFAULT_NUM_RECORDS: int = 1000

# Degradation simulation parameters
DEGRADATION_RATE_RANGE: tuple = (0.5, 1.5)  # Component degradation speed variance
INITIAL_HEALTH_RANGE: tuple = (0.8, 1.2)    # Starting health level variance

# Noise parameters for different sensor categories
NOISE_STABLE_PARAMS: float = 0.02    # Low noise for stable params (rpm, bus_voltage)
NOISE_PRESSURE_PARAMS: float = 0.05  # Moderate noise for pressure sensors
NOISE_DEFAULT: float = 0.05           # Default noise level

# Probability of sudden pressure drop (simulating failure)
SUDDEN_DROP_PROBABILITY: float = 0.01

# Accelerated degradation factor after failure point
POST_FAILURE_DEGRADATION_FACTOR: float = 0.5


# ============================================================================
# MACHINE LEARNING MODEL PARAMETERS
# ============================================================================

# Random Forest parameters
RF_N_ESTIMATORS: int = 100
RF_RANDOM_STATE: int = 42

# LSTM parameters
LSTM_SEQUENCE_LENGTH: int = 10
LSTM_UNITS: int = 64
LSTM_LEARNING_RATE: float = 0.001
LSTM_EPOCHS: int = 20
LSTM_BATCH_SIZE: int = 16

# Model training parameters
TRAIN_TEST_SPLIT: float = 0.2
RANDOM_SEED: int = 42

# Prediction parameters
DEFAULT_CONFIDENCE: float = 0.85
DEFAULT_TIME_HORIZON: str = "100h"
HIGH_CONFIDENCE_THRESHOLD: float = 0.9


# ============================================================================
# FMEA (Failure Mode and Effects Analysis) PARAMETERS
# ============================================================================

# RPN thresholds for alerts
RPN_LOW_THRESHOLD: int = 50
RPN_MEDIUM_THRESHOLD: int = 100
RPN_HIGH_THRESHOLD: int = 200
RPN_CRITICAL_THRESHOLD: int = 300

# FMEA rating scales (1-10)
SEVERITY_MIN: int = 1
SEVERITY_MAX: int = 10
OCCURRENCE_MIN: int = 1
OCCURRENCE_MAX: int = 10
DETECTION_MIN: int = 1
DETECTION_MAX: int = 10


# ============================================================================
# FAA COMPLIANCE PARAMETERS
# ============================================================================

# AC 43-12C alignment flags
PILOT_ALLOWED_TASKS: List[str] = [
    "Visual Inspection",
    "Oil Level Check",
    "Tire Pressure Check",
    "Landing Light Replacement"
]

# Tasks requiring A&P certification
AP_REQUIRED_TASKS: List[str] = [
    "Engine Overhaul",
    "Propeller Replacement",
    "Avionics Installation",
    "Structural Repair"
]


# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Streamlit page configuration
PAGE_TITLE: str = "GA Predictive Maintenance Dashboard"
PAGE_LAYOUT: str = "wide"

# Auto-refresh settings
DEFAULT_REFRESH_INTERVAL: int = 10  # seconds
MAX_REFRESH_INTERVAL: int = 60      # seconds

# Chart settings
CHART_FIGURE_SIZE: tuple = (10, 5)
CHART_COLOR_PALETTE: str = "Blues"

# Dark mode colors
DARK_MODE_BG: str = "linear-gradient(135deg, #121212, #2c3e50)"
DARK_MODE_CARD_BG: str = "#1e272e"
DARK_MODE_TEXT: str = "#f1f1f1"
DARK_MODE_METRIC_BG: str = "#34495e"

# Light mode colors
LIGHT_MODE_BG: str = "linear-gradient(135deg, #e8f0f8, #ffffff)"
LIGHT_MODE_CARD_BG: str = "#ffffff"
LIGHT_MODE_TEXT: str = "#333333"
LIGHT_MODE_METRIC_BG: str = "#00796b"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log file settings
LOG_FILE: Path = LOG_DIR / "pdm_system.log"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"


# ============================================================================
# DATABASE QUERY LIMITS
# ============================================================================

# Maximum records to fetch in dashboard queries
MAX_PREDICTION_RECORDS: int = 100
MAX_SENSOR_RECORDS: int = 10000


# ============================================================================
# VALIDATION PARAMETERS
# ============================================================================

# Required fields in performance metrics JSON
REQUIRED_METRIC_FIELDS: List[str] = [
    "precision",
    "recall",
    "accuracy",
    "f1_score"
]

# Tail number validation pattern
TAIL_NUMBER_MIN_LENGTH: int = 2
TAIL_NUMBER_MAX_LENGTH: int = 10


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_path() -> str:
    """
    Get the database path as a string.

    Returns:
        str: Absolute path to the database file
    """
    return str(DATABASE_PATH.absolute())


def get_model_dir() -> str:
    
    # Get the model directory path as a string.   
    return str(MODEL_DIR.absolute())


def validate_paths() -> bool:

    # Validate that all critical paths exist and are accessible.    
    critical_paths = [DATABASE_DIR, MODEL_DIR]
    return all(path.exists() and path.is_dir() for path in critical_paths)    # bool: True if all paths are valid, False otherwise
