# General Aviation Predictive Maintenance System

A comprehensive end-to-end predictive maintenance (PdM) system for General Aviation, featuring synthetic data generation, machine learning models (Random Forest + LSTM), FMEA/RPN integration, FAA AC 43-12C compliance, and interactive dashboards.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Airbus Evaluation Framework Alignment](#airbus-evaluation-framework-alignment)
- [Technical Documentation](#technical-documentation)
- [Contributing](#contributing)

---

## Overview

This PdM system demonstrates a production-ready implementation of predictive maintenance for aircraft components, aligning with industry standards and best practices. The system integrates:

- **High-variability synthetic sensor data** simulating realistic operational behavior
- **SQLite database** with comprehensive schema (40+ tables, views, triggers)
- **ML pipeline** featuring Random Forest and LSTM models for RUL prediction
- **FMEA/RPN framework** for failure mode analysis
- **FAA AC 43-12C compliance** for preventive maintenance task mapping
- **Interactive Streamlit dashboards** for real-time monitoring

### Built For

This project was developed for technical interviews at aerospace companies (e.g., Airbus Data Science Engineer), demonstrating:

1. **Model Integration Pipeline** - End-to-end data flow from sensors to predictions
2. **Data Engineering** - Schema design, ETL, feature engineering
3. **Best Practices** - Code quality, documentation, modularity, logging
4. **Traceability** - FMEA, calibration, FAA alignment, model metadata

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                   │
│  ┌────────────────┐      ┌──────────────────────────┐          │
│  │  Synthetic     │      │   SQLite Database        │          │
│  │  Data Generator│─────▶│  - 40+ Tables            │          │
│  │  (Sensor Data) │      │  - Views & Triggers      │          │
│  └────────────────┘      │  - FMEA Integration      │          │
│                          └──────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ETL & ML PIPELINE                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ Extract  │──▶│ Validate │──▶│Transform │──▶│  Train   │   │
│  │  Sensor  │   │  Schema  │   │ Features │   │  Models  │   │
│  │   Data   │   │  & Range │   │ Sequences│   │ RF+LSTM  │   │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                       │          │
│                                     ┌─────────────────┘          │
│                                     ▼                            │
│                          ┌──────────────────┐                   │
│                          │   Predictions    │                   │
│                          │   → Database     │                   │
│                          └──────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DASHBOARD LAYER                                 │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Component     │  │  Model          │  │  Preventive     │ │
│  │  Health View   │  │  Monitoring     │  │  Tasks (FAA)    │ │
│  └────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 1. Synthetic Data Generation
- Simulates realistic sensor behavior: drift, spikes, decay, noise
- Variable sampling rates per sensor type
- Gradual and accelerated degradation patterns
- 10 key sensor parameters (CHT, oil pressure, RPM, voltage, etc.)

### 2. Database Design
- **Aircraft & Components**: Tail numbers, models, total hours
- **Sensor Data**: Time-series readings with health flags
- **Maintenance Logs**: Historical maintenance events
- **Predictions**: Model outputs with confidence, time horizons
- **FMEA**: Severity, Occurrence, Detection ratings → RPN
- **Preventive Tasks**: FAA AC 43-12A mapped maintenance actions
- **Views**: Precomputed analytics (components_needing_attention, engine_health_view)
- **Triggers**: Auto-calculate RPN, generate alerts, normalize dates

### 3. Machine Learning
- **Random Forest Regressor**: Baseline model for RUL prediction
- **LSTM Neural Network**: Sequential model for time-series patterns
- **Features**: 10 normalized sensor parameters
- **Target**: Remaining Useful Life (hours)
- **Metrics**: MAE, RMSE, R², logged to database

### 4. FMEA & RPN
- Failure modes mapped to components
- Severity × Occurrence × Detection = RPN
- Automatic alerts triggered by RPN thresholds
- Traceability from sensor anomaly → failure mode → maintenance task

### 5. FAA Compliance
- Preventive tasks aligned with AC 43-12A
- Pilot-allowed vs. A&P-required task classification
- Component failure modes linked to regulatory guidance

### 6. Interactive Dashboards
- **Main Dashboard**: Component health, RUL trends, critical alerts
- **Model Monitor**: Performance metrics, JSON validation
- **PdM Dashboard**: Multiple views (predictions, engine health, alerts)
- **Preventive Tasks**: Filtered task lists, CSV export

---

## Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/brewks/ga_maintenance.git
   cd ga_maintenance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database**
   ```bash
   # The database will auto-restore from full_pdm_seed.sql on first dashboard launch
   # Or manually restore:
   sqlite3 database/ga_maintenance.db < database/full_pdm_seed.sql
   ```

---

## Quick Start

### 1. Generate Synthetic Sensor Data
```bash
python data_generation/synthetic_sensor_generator.py
```

### 2. Train ML Models
```bash
python ml/train_models.py
```

### 3. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

Navigate to `http://localhost:8501` in your browser.

### 4. View PdM Dashboard
```bash
streamlit run dashboard/pages/pdm_dashboard.py
```

### 5. Monitor Models
```bash
streamlit run dashboard/pages/model_monitor.py
```

---

## Project Structure

```
ga_maintenance/
├── config/                      # Configuration and utilities
│   ├── settings.py              # Centralized constants and paths
│   ├── db_utils.py              # Database helper functions
│   └── logging_config.py        # Logging setup
│
├── data_generation/             # Synthetic data generation
│   └── synthetic_sensor_generator.py
│
├── ml/                          # Machine learning models
│   ├── models/                  # Saved model files (.pkl, .keras)
│   └── train_models.py          # Training pipeline
│
├── dashboard/                   # Streamlit web application
│   ├── app.py                   # Main dashboard entry point
│   └── pages/
│       ├── pdm_dashboard.py     # Predictive maintenance views
│       ├── model_monitor.py     # Model performance tracking
│       └── due_preventive_tasks.py  # FAA-aligned task list
│
├── database/                    # SQLite database and schema
│   ├── ga_maintenance.db        # Database file (generated)
│   ├── full_pdm_seed.sql        # Complete schema + seed data
│   ├── insert_sensor_data.sql   # Sensor data inserts
│   └── insert_maintenance_recommendations.sql
│
├── docs/                        # Documentation
│   ├── SCHEMA_OVERVIEW.md       # Database schema guide
│   ├── FMEA_DOCUMENTATION.md    # FMEA and RPN explanation
│   └── PIPELINE_OVERVIEW.md     # ETL and ML pipeline details
│
├── logs/                        # Application logs
│   └── pdm_system.log
│
├── backups/                     # Database backups
│
├── scripts/                     # Utility scripts
│   └── backup_rotation.sh       # Automated backup script
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Usage Guide

### Generating Synthetic Data

The synthetic data generator creates realistic sensor readings with degradation patterns:

```python
from data_generation import generate_synthetic_data
from config.settings import DATABASE_PATH, TOP_SENSOR_PARAMS

generate_synthetic_data(
    db_path=DATABASE_PATH,
    params=TOP_SENSOR_PARAMS,
    num_components=20,
    num_records=2000
)
```

**Key Parameters:**
- `num_components`: Number of aircraft components to simulate
- `num_records`: Number of time-series records per component per sensor
- Generates variable sampling rates, drift, spikes, and failures

### Training Models

The ML pipeline extracts data, creates sequences, trains models, and logs predictions:

```bash
python ml/train_models.py
```

**Pipeline Steps:**
1. Extract sensor data and component RUL from database
2. Clean and validate data (timestamps, ranges)
3. Pivot data to wide format (one row per component-timestamp)
4. Normalize features using MinMaxScaler
5. Create sequences for LSTM (default: 10 timesteps)
6. Train/test split (80/20)
7. Train Random Forest (last timestep features)
8. Train LSTM (full sequences)
9. Evaluate metrics (MAE, RMSE, R²)
10. Log predictions to `component_predictions` table

**Model Outputs:**
- `ml/models/rf_model_rul.pkl` - Random Forest model
- `ml/models/model_lstm_rul.keras` - LSTM model
- `ml/models/predictions_comparison.png` - Visualization

### Dashboard Features

#### Main Dashboard (`dashboard/app.py`)
- Component selector (tail number + name)
- Health status: condition, RUL, health score
- Critical alerts (confidence > 90%)
- RUL trend chart (Altair interactive)
- Model performance metrics display
- JSON metrics validator

#### PdM Dashboard (`dashboard/pages/pdm_dashboard.py`)
- **Views:**
  - Components Needing Attention
  - Dashboard Snapshot
  - Latest Predictions (with charts)
  - Engine Health Overview
- Confidence filtering
- CSV export
- Auto-refresh capability

#### Model Monitor (`dashboard/pages/model_monitor.py`)
- List all models with IDs, names, algorithms
- Detailed performance metrics per model
- Download metrics as JSON
- Custom JSON validator

#### Preventive Tasks (`dashboard/pages/due_preventive_tasks.py`)
- FAA AC 43-12A aligned task list
- Filter by system and tail number
- Task breakdown by system (bar chart)
- CSV export

---

## Airbus Evaluation Framework Alignment

This project directly addresses the four evaluation buckets for aerospace data science interviews:

### 1. Model Integration Pipeline ⭐⭐⭐
- **End-to-end data flow**: Sensors → DB → ETL → ML → Predictions → Dashboard
- **Automated pipeline**: `train_models.py` orchestrates extraction, training, evaluation, logging
- **Traceability**: Model metadata (ID, version, metrics) stored in `predictive_models` table
- **Prediction lineage**: `input_data_hash`, `feature_vector_summary`, `model_input_features` fields

### 2. Data Engineering ⭐⭐⭐
- **Schema design**: 40+ tables with proper constraints, foreign keys, indexes
- **ETL pipeline**: Extract (SQL), Validate (schema + domain), Transform (pivot, normalize, sequences), Load (predictions)
- **Feature engineering**: Rolling windows, degradation metrics, LSTM sequence generation
- **Data quality**: Triggers for date normalization, RPN calculation, alert generation

### 3. Best Practices ⭐⭐⭐
- **Code organization**: Modular packages (config, data_generation, ml, dashboard)
- **Configuration management**: Centralized `settings.py` eliminates hard-coded values
- **Type hints**: All functions use proper type annotations
- **Docstrings**: Comprehensive documentation for all modules and functions
- **Logging**: Structured logging via `logging_config.py` replaces print statements
- **Error handling**: Try-except blocks with proper logging
- **Version control**: Git-ready structure

### 4. Traceability ⭐⭐⭐
- **FMEA integration**: Severity, Occurrence, Detection → RPN → Alerts
- **Calibration tracking**: `sensor_calibration` and `sensor_health_metrics` tables
- **Model versioning**: `predictive_models` table with training date, data snapshot
- **FAA compliance**: Preventive tasks mapped to AC 43-12C references
- **Audit trail**: Maintenance logs, prediction timestamps, confidence levels

---

## Technical Documentation

For detailed technical information, see:

- **[Database Schema Overview](docs/SCHEMA_OVERVIEW.md)** - Table descriptions, relationships, triggers
- **[FMEA Documentation](docs/FMEA_DOCUMENTATION.md)** - Failure mode analysis, RPN calculations
- **[Pipeline Overview](docs/PIPELINE_OVERVIEW.md)** - ETL process, feature engineering, model architecture

---

## Configuration

All system settings are centralized in `config/settings.py`:

- **Paths**: Database, models, logs, backups
- **Sensor parameters**: List of monitored parameters, units, thresholds
- **Synthetic data**: Noise levels, degradation rates, sampling intervals
- **ML hyperparameters**: RF estimators, LSTM units, epochs, batch size
- **FMEA thresholds**: RPN levels for alerts
- **Dashboard settings**: Colors, refresh rates, chart sizes

**Example customization:**
```python
# config/settings.py
LSTM_EPOCHS = 50  # Increase training epochs
DEFAULT_NUM_COMPONENTS = 50  # Generate more synthetic components
HIGH_CONFIDENCE_THRESHOLD = 0.95  # Stricter alert threshold
```

---

## Database Schema Highlights

### Core Tables
- `aircraft`: Aircraft registration, model, total hours, predictive status
- `components`: Component health, RUL, last health score
- `sensor_data`: Time-series sensor readings with health flags
- `maintenance_logs`: Historical maintenance events
- `predictive_models`: Model metadata, performance metrics JSON
- `component_predictions`: RUL predictions with confidence, time horizons
- `fmea_ratings`: Severity, Occurrence, Detection per failure mode
- `preventive_tasks`: FAA AC 43-12A maintenance task library
- `pdm_task_mapping`: Links failure modes → preventive tasks

### Key Views
- `components_needing_attention`: Components with RUL < threshold or high RPN
- `engine_health_view`: Engine-specific health metrics
- `sensor_health_overview`: Calibration status per sensor
- `dashboard_snapshot_view`: Unified view for dashboards
- `due_preventive_tasks`: Actionable maintenance tasks

### Triggers
- **Date normalization**: Ensures consistent YYYY-MM-DD format
- **RPN calculation**: Auto-computes Severity × Occurrence × Detection
- **Alert generation**: Creates predictive alerts from model outputs

---

## FAA AC 43-12C Compliance

Advisory Circular 43-12C provides guidance on preventive maintenance tasks that pilots can perform versus those requiring A&P certification.

### Implementation
1. **Task Library**: `preventive_tasks` table contains AC 43-12A tasks
2. **Pilot-Allowed Flag**: Boolean field indicating pilot eligibility
3. **A&P Required**: Tasks requiring certified mechanic
4. **Failure Mode Mapping**: `pdm_task_mapping` links predictions to tasks
5. **Dashboard Integration**: `due_preventive_tasks` page displays actionable items

**Example:**
- **Visual Inspection** → Pilot-allowed
- **Engine Overhaul** → A&P required

---

## Performance Considerations

### Database
- **Indexes**: Created on frequently queried columns (component_id, timestamp)
- **Views**: Precomputed for dashboard performance
- **Triggers**: Lightweight, execute only on relevant events

### ML Pipeline
- **Batch processing**: Sequences generated in batches to manage memory
- **Normalization**: MinMaxScaler fits once, transforms efficiently
- **Model storage**: Joblib for RF (fast), Keras format for LSTM

### Dashboard
- **Caching**: `@st.cache_data` decorators reduce database queries
- **TTL**: Cache expires after 60 seconds for near-real-time updates
- **Lazy loading**: Data loaded only when view is selected

---

## Logging

All components use structured logging:

```python
from config.logging_config import setup_logger

logger = setup_logger(__name__)
logger.info("System initialized")
logger.warning("Low confidence prediction")
logger.error("Database connection failed", exc_info=True)
```

**Log file**: `logs/pdm_system.log`

**Format**:
```
2025-11-23 14:30:15 - ml.train_models - INFO - Training LSTM model...
2025-11-23 14:30:45 - ml.train_models - INFO - LSTM - MAE: 12.34, RMSE: 18.56, R²: 0.8723
```

---

## Testing

### Validate Installation
```bash
# Test database connection
python -c "from config.db_utils import check_database_exists; print('DB exists:', check_database_exists())"

# Test data generation
python data_generation/synthetic_sensor_generator.py

# Test ML pipeline
python ml/train_models.py

# Launch dashboard
streamlit run dashboard/app.py
```

### Expected Outputs
- Synthetic data: ~100k+ sensor records inserted
- ML training: Models saved, metrics logged, predictions inserted
- Dashboard: No errors, data displays correctly

---

## Troubleshooting

### Database Not Found
- Check `database/ga_maintenance.db` exists
- Manually restore: `sqlite3 database/ga_maintenance.db < database/full_pdm_seed.sql`

### Import Errors
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Add project root to PYTHONPATH if needed

### Dashboard Won't Start
- Check port 8501 is available
- Verify Streamlit installed: `streamlit --version`

### Model Training Fails
- Ensure sufficient data in `sensor_data` table
- Check sequence length doesn't exceed available timesteps
- Verify component RUL values are not null

---

## Contributing

This is a portfolio project for technical interviews. For suggestions or improvements:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request

---

## License

This project is provided as-is for educational and interview purposes.

---

## Contact

- **Author**: Ndubuisi O. Chibuogwu
- **GitHub**: https://github.com/brewks/ga_maintenance
- **Purpose**: General Aviation Use & Airbus Data Science Engineer Technical Interview

---

## Acknowledgments

- **FAA AC 43-12A**: Preventive maintenance guidance
- **FMEA Best Practices**: Automotive and aerospace FMEA standards
- **Airbus Interview Framework**: Model Integration, Data Engineering, Best Practices, Traceability

---

**Last Updated**: November 2025
**Version**: 1.0.0
