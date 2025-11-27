# Database Schema Overview

This document provides a comprehensive overview of the GA Predictive Maintenance System database schema.

---

## Database Technology

**SQLite 3** - Lightweight, serverless, self-contained SQL database engine.

**Location**: `database/ga_maintenance.db`

---

## Schema Summary

- **40+ tables**: Core data, analytics, FMEA, FAA compliance
- **10+ views**: Precomputed analytics for dashboards
- **Multiple triggers**: Auto-calculate RPN, normalize dates, generate alerts
- **Indexes**: Optimized for time-series queries

---

## Core Tables

### 1. `aircraft`
Stores aircraft registration and operational data.

**Columns:**
- `tail_number` (TEXT, PRIMARY KEY): FAA registration (e.g., "N12345")
- `model` (TEXT): Aircraft model (e.g., "Cessna 172")
- `manufacturer` (TEXT): Manufacturer name
- `total_hours` (REAL): Total flight hours
- `last_annual_date` (TEXT): Date of last annual inspection
- `predictive_status` (TEXT): Current PdM status (normal, monitoring, attention_needed, maintenance_required)
- `last_predictive_analysis` (TEXT): Timestamp of last ML prediction

**Purpose**: Central registry of aircraft in the fleet.

---

### 2. `components`
Tracks individual aircraft components and their health.

**Columns:**
- `component_id` (INTEGER, PRIMARY KEY)
- `tail_number` (TEXT, FOREIGN KEY → aircraft)
- `name` (TEXT): Component name (e.g., "Engine", "Landing Gear")
- `component_type` (TEXT): Category (engine, hydraulic, electrical, etc.)
- `condition` (TEXT): Current condition (excellent, good, fair, poor, critical)
- `remaining_useful_life` (REAL): Predicted RUL in hours
- `last_health_score` (REAL): Latest health score (0-100)
- `installation_date` (TEXT): When component was installed
- `last_maintenance` (TEXT): Date of last maintenance event

**Purpose**: Tracks component-level health and RUL predictions.

---

### 3. `sensor_data`
Time-series sensor readings from aircraft components.

**Columns:**
- `sensor_id` (INTEGER, PRIMARY KEY)
- `tail_number` (TEXT): Aircraft identifier
- `component_id` (INTEGER, FOREIGN KEY → components)
- `parameter` (TEXT): Sensor parameter name (e.g., "oil_press", "cht", "rpm")
- `value` (REAL): Sensor reading value
- `unit` (TEXT): Unit of measurement (psi, °C, rpm, etc.)
- `timestamp` (TEXT): Reading timestamp (YYYY-MM-DD HH:MM:SS)
- `sensor_health` (INTEGER): Sensor health flag (0 = healthy, 1 = unhealthy)

**Purpose**: Stores raw sensor telemetry for ML training and monitoring.

**Indexes:**
- `idx_sensor_component_time` on (`component_id`, `timestamp`)
- `idx_sensor_parameter` on (`parameter`)

---

### 4. `maintenance_logs`
Historical maintenance events.

**Columns:**
- `log_id` (INTEGER, PRIMARY KEY)
- `tail_number` (TEXT, FOREIGN KEY → aircraft)
- `component_id` (INTEGER, FOREIGN KEY → components)
- `event_date` (TEXT): When maintenance occurred
- `event_type` (TEXT): Type of maintenance (inspection, repair, replacement, etc.)
- `description` (TEXT): Detailed description
- `technician` (TEXT): Who performed the work
- `cost` (REAL): Cost of maintenance

**Purpose**: Audit trail of maintenance history.

---

### 5. `predictive_models`
Metadata for trained ML models.

**Columns:**
- `model_id` (INTEGER, PRIMARY KEY)
- `model_name` (TEXT): Human-readable name
- `model_type` (TEXT): Algorithm (Random Forest, LSTM, etc.)
- `training_date` (TEXT): When model was trained
- `performance_metrics` (TEXT): JSON with precision, recall, accuracy, f1_score
- `data_snapshot_id` (TEXT): Identifier for training data version
- `model_version` (TEXT): Version string

**Purpose**: Tracks model lineage and performance for traceability.

**Example performance_metrics JSON:**
```json
{
  "precision": 0.92,
  "recall": 0.89,
  "accuracy": 0.91,
  "f1_score": 0.90
}
```

---

### 6. `component_predictions`
ML model predictions for component RUL.

**Columns:**
- `prediction_id` (INTEGER, PRIMARY KEY)
- `component_id` (INTEGER, FOREIGN KEY → components)
- `model_id` (INTEGER, FOREIGN KEY → predictive_models)
- `prediction_type` (TEXT): Type (remaining_life, failure, anomaly)
- `predicted_value` (REAL): Predicted RUL in hours
- `confidence` (REAL): Prediction confidence (0.0-1.0)
- `time_horizon` (TEXT): Forecast horizon (e.g., "100h")
- `explanation` (TEXT): Human-readable explanation
- `prediction_time` (TEXT): When prediction was made
- `input_data_hash` (TEXT): Hash of input data for reproducibility
- `feature_vector_summary` (TEXT): Summary of features used
- `model_input_features` (TEXT): List of feature names
- `failure_mode` (TEXT): Associated failure mode (links to FMEA)

**Purpose**: Stores all model predictions with full traceability.

---

## FMEA Tables

### 7. `fmea_ratings`
Failure Mode and Effects Analysis ratings.

**Columns:**
- `fmea_id` (INTEGER, PRIMARY KEY)
- `component_id` (INTEGER, FOREIGN KEY → components)
- `failure_mode` (TEXT): Description of failure mode
- `severity` (INTEGER): Severity rating (1-10)
- `occurrence` (INTEGER): Occurrence rating (1-10)
- `detection` (INTEGER): Detection rating (1-10)
- `rpn` (INTEGER): Risk Priority Number (Severity × Occurrence × Detection)

**Purpose**: Systematically evaluates failure risks.

**RPN Calculation (Trigger):**
```sql
CREATE TRIGGER calculate_rpn
AFTER INSERT ON fmea_ratings
BEGIN
  UPDATE fmea_ratings
  SET rpn = NEW.severity * NEW.occurrence * NEW.detection
  WHERE fmea_id = NEW.fmea_id;
END;
```

---

### 8. `failure_mode_rules`
Maps failure modes to detection rules.

**Columns:**
- `rule_id` (INTEGER, PRIMARY KEY)
- `failure_mode` (TEXT): Failure mode name
- `sensor_parameter` (TEXT): Sensor to monitor
- `threshold_value` (REAL): Threshold for detection
- `operator` (TEXT): Comparison operator (<, >, =, etc.)

**Purpose**: Links sensor anomalies to specific failure modes.

---

## FAA Compliance Tables

### 9. `preventive_tasks`
Library of FAA AC 43-12C preventive maintenance tasks.

**Columns:**
- `task_id` (INTEGER, PRIMARY KEY)
- `task_name` (TEXT): Task name
- `description` (TEXT): Detailed description
- `ac_43_12c_reference` (TEXT): Specific AC 43-12C section
- `pilot_allowed` (INTEGER): 1 if pilot can perform, 0 if A&P required
- `estimated_time` (REAL): Estimated completion time (hours)

**Purpose**: Maps PdM predictions to regulatory-compliant maintenance actions.

---

### 10. `pdm_task_mapping`
Links failure modes to preventive tasks.

**Columns:**
- `mapping_id` (INTEGER, PRIMARY KEY)
- `failure_mode` (TEXT, FOREIGN KEY → failure_mode_rules)
- `task_id` (INTEGER, FOREIGN KEY → preventive_tasks)

**Purpose**: Creates actionable maintenance recommendations from predictions.

---

## Analytics Views

### `components_needing_attention`
Components requiring maintenance based on RUL or RPN.

**Logic:**
```sql
SELECT c.component_id, c.tail_number, c.name,
       c.remaining_useful_life, f.rpn,
       CASE
         WHEN c.remaining_useful_life < 50 THEN 'LOW_RUL'
         WHEN f.rpn > 200 THEN 'HIGH_RPN'
         ELSE 'MONITORING'
       END AS attention_reason
FROM components c
LEFT JOIN fmea_ratings f ON c.component_id = f.component_id
WHERE c.remaining_useful_life < 100 OR f.rpn > 100;
```

---

### `engine_health_view`
Engine-specific health metrics.

**Combines:**
- Component health scores
- Latest sensor readings (oil pressure, CHT, RPM)
- Prediction confidence
- Maintenance history

---

### `sensor_health_overview`
Calibration status and health flags for all sensors.

**Columns:**
- Sensor ID and parameter
- Calibration date and next due date
- Health status (healthy, degraded, faulty)
- Outlier detection results

---

### `dashboard_snapshot_view`
Unified view for dashboard displays.

**Combines:**
- Aircraft + component info
- Latest predictions
- FMEA ratings
- Sensor health status
- Due maintenance tasks

---

### `due_preventive_tasks`
Actionable tasks based on current predictions.

**Logic:**
```sql
SELECT a.tail_number, c.name AS component,
       pt.task_name, pt.ac_43_12c_reference,
       pt.pilot_allowed, p.confidence,
       p.prediction_time AS timestamp
FROM component_predictions p
JOIN components c ON p.component_id = c.component_id
JOIN aircraft a ON c.tail_number = a.tail_number
JOIN pdm_task_mapping m ON p.failure_mode = m.failure_mode
JOIN preventive_tasks pt ON m.task_id = pt.task_id
WHERE p.confidence > 0.8 AND p.prediction_type = 'failure';
```

---

## Key Triggers

### 1. Date Normalization
Ensures all dates are stored as YYYY-MM-DD.

```sql
CREATE TRIGGER normalize_maintenance_dates
BEFORE INSERT ON maintenance_logs
BEGIN
  UPDATE maintenance_logs
  SET event_date = date(NEW.event_date)
  WHERE log_id = NEW.log_id;
END;
```

---

### 2. Automatic RPN Calculation
Computes RPN when FMEA ratings are inserted/updated.

```sql
CREATE TRIGGER calculate_rpn
AFTER INSERT ON fmea_ratings
BEGIN
  UPDATE fmea_ratings
  SET rpn = NEW.severity * NEW.occurrence * NEW.detection
  WHERE fmea_id = NEW.fmea_id;
END;
```

---

### 3. Alert Generation
Creates predictive alerts from model outputs.

```sql
CREATE TRIGGER generate_alerts
AFTER INSERT ON component_predictions
WHEN NEW.confidence > 0.9 AND NEW.prediction_type = 'failure'
BEGIN
  INSERT INTO alerts (component_id, alert_type, severity, message, timestamp)
  VALUES (NEW.component_id, 'PREDICTIVE_FAILURE', 'HIGH',
          'Predicted failure in ' || NEW.time_horizon, datetime('now'));
END;
```

---

## Entity-Relationship Diagram (Simplified)

```
┌──────────────┐
│   aircraft   │
│ (tail_number)│
└──────┬───────┘
       │ 1:N
       ▼
┌──────────────────┐
│   components     │◄──────┐
│  (component_id)  │       │
└────────┬─────────┘       │ N:1
         │ 1:N             │
         ▼                 │
┌──────────────────┐       │
│  sensor_data     │       │
└──────────────────┘       │
                           │
┌──────────────────┐       │
│predictive_models │       │
└────────┬─────────┘       │
         │ 1:N             │
         ▼                 │
┌────────────────────────┐ │
│ component_predictions  │─┘
└────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌────────────────────┐
│  failure_mode    │────►│  pdm_task_mapping  │
│    _rules        │ N:M │                    │
└──────────────────┘     └──────────┬─────────┘
                                    │ N:1
                                    ▼
                         ┌────────────────────┐
                         │ preventive_tasks   │
                         └────────────────────┘
```

---

## Performance Optimizations

1. **Indexes on time-series queries**:
   - `idx_sensor_component_time` for fast sensor data retrieval
   - `idx_predictions_component` for quick prediction lookup

2. **Materialized views** (via precomputed views):
   - `dashboard_snapshot_view` reduces JOIN overhead

3. **Triggers**:
   - Lightweight, execute only on specific events
   - Pre-compute RPN to avoid runtime calculation

---

## Data Integrity Constraints

- **Foreign keys** ensure referential integrity
- **CHECK constraints** enforce valid value ranges (e.g., severity 1-10)
- **NOT NULL** constraints prevent incomplete records
- **UNIQUE** constraints on tail numbers, sensor IDs

---

## For More Information

- See [FMEA_DOCUMENTATION.md](FMEA_DOCUMENTATION.md) for FMEA/RPN details
- See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for ETL and ML pipeline
- See main [README.md](../README.md) for system overview

---

**Last Updated**: November 2025
