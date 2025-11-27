# ML Pipeline Overview

This document describes the complete ETL (Extract, Transform, Load) and ML training pipeline for the GA Predictive Maintenance System.

---

## Pipeline Architecture

The pipeline follows a classic ETL → Train → Predict → Load workflow:

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   EXTRACT   │─────▶│  TRANSFORM   │─────▶│    TRAIN     │
│  Sensor Data│      │   Features   │      │   Models     │
└─────────────┘      └──────────────┘      └──────┬───────┘
                                                   │
                                                   ▼
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   MONITOR   │◀─────│     LOAD     │◀─────│   PREDICT    │
│  Dashboard  │      │  Predictions │      │     RUL      │
└─────────────┘      └──────────────┘      └──────────────┘
```

---

## Phase 1: Extract

**Module**: `ml/train_models.py` - Functions: `extract_sensor_data()`, `extract_components_rul()`

### 1.1 Extract Sensor Data

```python
def extract_sensor_data() -> pd.DataFrame:
    query = "SELECT * FROM sensor_data"
    df = execute_query(query)
    return df
```

**Output**: DataFrame with columns:
- `sensor_id`: Unique sensor reading ID
- `tail_number`: Aircraft identifier
- `component_id`: Component being monitored
- `parameter`: Sensor type (oil_press, cht, rpm, etc.)
- `value`: Sensor reading value
- `unit`: Unit of measurement
- `timestamp`: When reading was taken
- `sensor_health`: Health flag (0=healthy, 1=unhealthy)

**Size**: Typically 100k+ rows (10 components × 10 parameters × 1000 timesteps)

### 1.2 Extract Component RUL

```python
def extract_components_rul() -> pd.DataFrame:
    query = "SELECT component_id, remaining_useful_life FROM components"
    df = execute_query(query)
    return df
```

**Output**: Target variable (RUL) for each component.

---

## Phase 2: Validate & Clean

**Module**: `ml/train_models.py` - Function: `clean_sensor_data()`

### 2.1 Data Validation

```python
def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Filter for top parameters
    df = df[df['parameter'].isin(TOP_SENSOR_PARAMS)]

    # Remove duplicates
    df = df.drop_duplicates(subset=['component_id', 'timestamp', 'parameter'])

    return df
```

**Validation Steps:**
1. **Timestamp validation**: Remove records with invalid timestamps
2. **Parameter filtering**: Keep only the 10 key sensor parameters
3. **Deduplication**: Remove duplicate readings
4. **Range checking**: Implicitly handled (values outside realistic ranges are outliers but not removed)

**Output**: Clean sensor data ready for transformation.

---

## Phase 3: Transform Features

**Module**: `ml/train_models.py` - Functions: `pivot_sensor_data()`, `normalize_features()`, `create_sequences()`

### 3.1 Pivot to Wide Format

**Input**: Long format (one row per sensor reading)

| component_id | timestamp | parameter | value |
|--------------|-----------|-----------|-------|
| 1 | 2025-01-01 10:00 | oil_press | 45.2 |
| 1 | 2025-01-01 10:00 | cht | 180.5 |
| 1 | 2025-01-01 10:00 | rpm | 2450 |

**Output**: Wide format (one row per component-timestamp, one column per parameter)

| component_id | timestamp | oil_press | cht | rpm | ... |
|--------------|-----------|-----------|-----|-----|-----|
| 1 | 2025-01-01 10:00 | 45.2 | 180.5 | 2450 | ... |

```python
def pivot_sensor_data(df: pd.DataFrame) -> pd.DataFrame:
    pivoted = df.pivot_table(
        index=['component_id', 'timestamp'],
        columns='parameter',
        values='value'
    ).sort_index().reset_index()

    # Fill missing values
    pivoted.ffill(inplace=True)
    pivoted.bfill(inplace=True)
    pivoted = pivoted.dropna(subset=TOP_SENSOR_PARAMS)

    return pivoted
```

**Why**: ML models require features in columns, not rows.

### 3.2 Normalize Features

```python
def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler
```

**Transformation**: All sensor values scaled to [0, 1] range.

**Why**:
- Neural networks train better with normalized inputs
- Prevents features with large magnitudes from dominating
- Ensures all features contribute equally

**Example**:
- Oil pressure: 25-50 psi → normalized to 0.0-1.0
- RPM: 1000-3000 rpm → normalized to 0.0-1.0

### 3.3 Create Sequences (for LSTM)

```python
def create_sequences(
    pivoted_df: pd.DataFrame,
    components_df: pd.DataFrame,
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # For each component, create sliding window sequences
    for comp_id in pivoted_df['component_id'].unique():
        comp_data = pivoted_df[pivoted_df['component_id'] == comp_id].sort_values('timestamp')
        for i in range(len(comp_data) - seq_length + 1):
            sequence = comp_data[TOP_SENSOR_PARAMS].iloc[i:i+seq_length].values
            x_seq.append(sequence)
            y_seq.append(rul_value)

    return x_seq, y_seq, comp_ids
```

**Output**: 3D array of shape `(num_sequences, seq_length, num_features)`

**Example Sequence** (seq_length=10):
```
[
  [oil_press_t0, cht_t0, rpm_t0, ...],   # Timestep 0
  [oil_press_t1, cht_t1, rpm_t1, ...],   # Timestep 1
  ...
  [oil_press_t9, cht_t9, rpm_t9, ...],   # Timestep 9
]
→ Target: RUL = 150 hours
```

**Why**: LSTM models learn temporal patterns from sequences.

---

## Phase 4: Train Models

**Module**: `ml/train_models.py` - Functions: `train_random_forest()`, `train_lstm()`

### 4.1 Train/Test Split

```python
x_train, x_test, y_train, y_test, comp_train, comp_test = train_test_split(
    x_seq, y_seq, comp_ids,
    test_size=0.2,
    random_state=42
)
```

**Split**: 80% training, 20% testing

### 4.2 Random Forest Training

**Input**: Last timestep of each sequence (2D array)

```python
def train_random_forest(...):
    # Extract last timestep
    x_rf_train = x_train[:, -1, :]  # Shape: (num_samples, num_features)

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(x_rf_train, y_train_clean)

    # Evaluate
    y_pred = rf.predict(x_rf_test) * y_max
    mae = mean_absolute_error(y_test * y_max, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test * y_max, y_pred))
    r2 = r2_score(y_test * y_max, y_pred)

    # Save model
    joblib.dump(rf, "ml/models/rf_model_rul.pkl")

    return rf, metrics
```

**Architecture**:
- **Algorithm**: Random Forest (ensemble of decision trees)
- **Estimators**: 100 trees
- **Features**: 10 sensor parameters (last timestep only)
- **Target**: Normalized RUL

**Why Random Forest**:
- Fast training
- Handles non-linear relationships
- Robust to outliers
- Good baseline model

### 4.3 LSTM Training

**Input**: Full sequences (3D array)

```python
def train_lstm(...):
    # Build model
    model = Sequential([
        Input(shape=(10, 10)),  # (seq_length, num_features)
        LSTM(64),               # 64 LSTM units
        Dense(1)                # Single output (RUL)
    ])

    model.compile(optimizer=Adam(0.001), loss='mse')

    # Train
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(x_test, y_test)
    )

    # Evaluate
    y_pred = model.predict(x_test).flatten() * y_max
    # Calculate metrics...

    # Save model
    model.save("ml/models/model_lstm_rul.keras")

    return model, metrics
```

**Architecture**:
- **Input**: Sequences of 10 timesteps × 10 features
- **LSTM Layer**: 64 units (learns temporal patterns)
- **Dense Layer**: 1 output (RUL prediction)
- **Optimizer**: Adam with learning rate 0.001
- **Loss**: Mean Squared Error (MSE)

**Why LSTM**:
- Captures temporal dependencies
- Learns degradation patterns over time
- Better for time-series forecasting than static models

**Training Metrics Logged**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)
- Training/validation loss curves

---

## Phase 5: Predict

**Module**: `ml/train_models.py` - Function: `main()` (prediction section)

### 5.1 Generate Predictions

```python
# Random Forest predictions
y_rf_pred = rf_model.predict(x_test[:, -1, :]) * y_max

# LSTM predictions
y_lstm_pred = lstm_model.predict(x_test).flatten() * y_max
```

**Output**: RUL predictions for test set components.

### 5.2 Denormalize

RUL values were normalized to [0, 1] for training. Denormalize for real-world interpretation:

```python
y_pred_denormalized = y_pred_normalized * y_max
```

**Example**:
- Normalized prediction: 0.65
- y_max = 500 hours
- **Actual RUL: 0.65 × 500 = 325 hours**

---

## Phase 6: Load Predictions

**Module**: `ml/train_models.py` - Function: `log_predictions_to_database()`

### 6.1 Clear Old Predictions

```python
delete_query = "DELETE FROM component_predictions WHERE model_id = ?"
execute_write(delete_query, (model_id,))
```

### 6.2 Insert New Predictions

```python
def log_predictions_to_database(component_ids, predictions, model_id=99, confidence=0.85, time_horizon="100h"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    insert_query = """
        INSERT INTO component_predictions (
            component_id, model_id, prediction_type, predicted_value,
            confidence, time_horizon, explanation, prediction_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    execute_many(insert_query, records)
```

**Database Record Example**:

| component_id | model_id | prediction_type | predicted_value | confidence | time_horizon | prediction_time |
|--------------|----------|-----------------|-----------------|------------|--------------|-----------------|
| 42 | 99 | remaining_life | 325.4 | 0.85 | 100h | 2025-01-15 14:30:00 |

---

## Phase 7: Monitor (Dashboard)

**Module**: `dashboard/app.py`, `dashboard/pages/pdm_dashboard.py`

### 7.1 Query Predictions

```python
@st.cache_data(ttl=60)
def load_predictions_data():
    query = """
        SELECT * FROM component_predictions
        ORDER BY prediction_time DESC
    """
    return execute_query(query)
```

### 7.2 Visualize

- **Component Health Cards**: Show RUL, condition, health score
- **RUL Trend Charts**: Altair line charts of RUL over time
- **Confidence Scatter Plots**: RUL vs Component ID sized by confidence
- **Critical Alerts**: Highlight predictions with confidence > 90%

### 7.3 Link to FMEA

```python
# Dashboard automatically joins predictions with FMEA ratings
query = """
    SELECT p.component_id, p.predicted_value, p.confidence,
           f.severity, f.occurrence, f.detection, f.rpn
    FROM component_predictions p
    LEFT JOIN fmea_ratings f ON p.component_id = f.component_id
    WHERE p.confidence > 0.8
    ORDER BY f.rpn DESC;
"""
```

**Result**: Prioritized list of components by RPN.

---

## Pipeline Configuration

All pipeline parameters are centralized in `config/settings.py`:

```python
# Sensor parameters
TOP_SENSOR_PARAMS = ['cht', 'fuel_flow', 'rpm', 'manifold_press', ...]

# ML parameters
LSTM_SEQUENCE_LENGTH = 10
LSTM_UNITS = 64
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 16

# Train/test split
TRAIN_TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Prediction parameters
DEFAULT_CONFIDENCE = 0.85
DEFAULT_TIME_HORIZON = "100h"
```

**Benefit**: Easy to experiment with different hyperparameters without modifying code.

---

## Pipeline Execution

**Command**:
```bash
python ml/train_models.py
```

**Output**:
```
2025-01-15 14:00:00 - ml.train_models - INFO - Starting ML Model Training Pipeline
2025-01-15 14:00:05 - ml.train_models - INFO - Extracted 125,430 sensor records
2025-01-15 14:00:10 - ml.train_models - INFO - Cleaned sensor data: 125,430 → 118,562 records
2025-01-15 14:00:15 - ml.train_models - INFO - Pivoted data shape: (11,856, 12)
2025-01-15 14:00:20 - ml.train_models - INFO - Created 11,756 sequences from 10 components
2025-01-15 14:00:25 - ml.train_models - INFO - Training Random Forest model...
2025-01-15 14:00:35 - ml.train_models - INFO - Random Forest - MAE: 12.34, RMSE: 18.56, R²: 0.8723
2025-01-15 14:00:40 - ml.train_models - INFO - Training LSTM model...
Epoch 1/20
588/588 ━━━━━━━━━━━━━━━━━━━━ 15s 25ms/step - loss: 0.0234 - val_loss: 0.0189
...
Epoch 20/20
588/588 ━━━━━━━━━━━━━━━━━━━━ 14s 24ms/step - loss: 0.0045 - val_loss: 0.0052
2025-01-15 14:06:20 - ml.train_models - INFO - LSTM - MAE: 8.92, RMSE: 12.34, R²: 0.9234
2025-01-15 14:06:25 - ml.train_models - INFO - Successfully logged 2,351 predictions
2025-01-15 14:06:30 - ml.train_models - INFO - ML Model Training Pipeline Complete
```

---

## Feature Engineering Summary

| Feature Type | Description | Example |
|--------------|-------------|---------|
| **Raw Sensors** | Direct sensor readings | oil_press = 45.2 psi |
| **Normalized** | Scaled to [0, 1] | oil_press_norm = 0.73 |
| **Sequences** | Sliding windows of timesteps | [t-9, t-8, ..., t-1, t] |
| **Derived** (future) | Statistical features | oil_press_mean_10, oil_press_std_10 |

---

## Model Comparison

| Metric | Random Forest | LSTM | Winner |
|--------|---------------|------|--------|
| **MAE** | 12.34 hours | 8.92 hours | LSTM ✓ |
| **RMSE** | 18.56 hours | 12.34 hours | LSTM ✓ |
| **R²** | 0.8723 | 0.9234 | LSTM ✓ |
| **Training Time** | 10 seconds | 6 minutes | RF ✓ |
| **Interpretability** | High | Low | RF ✓ |

**Conclusion**: LSTM is used for predictions due to superior accuracy. RF serves as a fast baseline.

---

## Pipeline Extensibility

### Adding New Features

1. **Define in config**:
   ```python
   # config/settings.py
   NEW_SENSOR_PARAMS = TOP_SENSOR_PARAMS + ['vibration', 'fuel_flow_rate']
   ```

2. **Update extraction**:
   ```python
   # ml/train_models.py
   from config.settings import NEW_SENSOR_PARAMS
   df = df[df['parameter'].isin(NEW_SENSOR_PARAMS)]
   ```

3. **Retrain**:
   ```bash
   python ml/train_models.py
   ```

### Adding New Models

Create new training function in `ml/train_models.py`:

```python
def train_xgboost(...):
    import xgboost as xgb
    model = xgb.XGBRegressor(...)
    model.fit(x_train, y_train)
    # ... rest of training logic
```

---

## For More Information

- See [SCHEMA_OVERVIEW.md](SCHEMA_OVERVIEW.md) for database details
- See [FMEA_DOCUMENTATION.md](FMEA_DOCUMENTATION.md) for risk assessment
- See main [README.md](../README.md) for system overview

---

**Last Updated**: November 2025
