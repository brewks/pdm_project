# FMEA and RPN Documentation

This document explains the Failure Mode and Effects Analysis (FMEA) implementation and Risk Priority Number (RPN) calculation in the GA Predictive Maintenance System.

---

## What is FMEA?

**Failure Mode and Effects Analysis (FMEA)** is a systematic method for:
1. Identifying potential failure modes in a system
2. Assessing the risk associated with each failure mode
3. Prioritizing corrective actions

FMEA is widely used in automotive, aerospace, and manufacturing industries to improve reliability and safety.

---

## FMEA in GA PdM System

Our system integrates FMEA with ML predictions to create a comprehensive risk assessment framework:

```
Sensor Anomaly → ML Prediction → Failure Mode → FMEA Rating → RPN → Maintenance Task
```

**Example Flow:**
1. Oil pressure sensor reads 25 psi (below threshold of 30 psi)
2. LSTM model predicts "Oil System Failure" with 92% confidence
3. FMEA rating for "Oil System Failure": Severity=8, Occurrence=6, Detection=4
4. RPN = 8 × 6 × 4 = 192 (High priority)
5. System recommends "Inspect oil system" task from AC 43-12C

---

## RPN Calculation

**Risk Priority Number (RPN)** = Severity × Occurrence × Detection

### 1. Severity (SEV)
**Definition**: How serious is the effect of the failure on safety, performance, or operation?

| Rating | Description | Example |
|--------|-------------|---------|
| 10 | Catastrophic | Total loss of aircraft control, fatalities |
| 9 | Very High | Major damage, serious injuries |
| 8 | High | Significant damage, minor injuries |
| 7 | Moderate-High | System failure requiring immediate landing |
| 6 | Moderate | Degraded performance, landing advisable |
| 5 | Moderate-Low | Minor performance loss |
| 4 | Low | Slight annoyance to operator |
| 3 | Very Low | Minor defect noticed by some operators |
| 2 | Minimal | Minor defect noticed by discriminating operators |
| 1 | None | No effect |

**Example:**
- Engine oil system failure → Severity = 8 (can cause engine seizure)
- Landing light burnt out → Severity = 3 (minor inconvenience)

---

### 2. Occurrence (OCC)
**Definition**: How frequently does this failure mode occur?

| Rating | Description | Probability | Example Frequency |
|--------|-------------|-------------|-------------------|
| 10 | Very High | ≥ 1 in 2 | More than once per 2 flights |
| 9 | High | 1 in 3 | Once per 3 flights |
| 8 | High | 1 in 8 | Once per 8 flights |
| 7 | Moderate-High | 1 in 20 | Once per 20 flights |
| 6 | Moderate | 1 in 80 | Once per 80 flights |
| 5 | Moderate-Low | 1 in 400 | Once per 400 flight hours |
| 4 | Low | 1 in 2,000 | Once per 2,000 flight hours |
| 3 | Very Low | 1 in 15,000 | Once per 15,000 flight hours |
| 2 | Remote | 1 in 150,000 | Rarely occurs |
| 1 | Nearly Impossible | < 1 in 1,500,000 | Never occurred in similar designs |

**Example:**
- Oil system contamination → Occurrence = 4 (happens occasionally)
- Engine catastrophic failure → Occurrence = 2 (very rare)

---

### 3. Detection (DET)
**Definition**: What is the likelihood that the failure will be detected before it causes harm?

| Rating | Description | Detection Capability |
|--------|-------------|----------------------|
| 10 | Absolutely Uncertain | No known controls to detect |
| 9 | Very Remote | Very remote chance of detection |
| 8 | Remote | Remote chance of detection |
| 7 | Very Low | Very low chance of detection |
| 6 | Low | Low chance of detection |
| 5 | Moderate | Moderate chance of detection |
| 4 | Moderately High | Moderately high chance of detection |
| 3 | High | High chance of detection |
| 2 | Very High | Very high chance of detection |
| 1 | Almost Certain | Detection almost certain (automated alarms) |

**Example:**
- Low oil pressure → Detection = 2 (sensor alarm is obvious)
- Slow fuel contamination → Detection = 7 (hard to detect until failure)

---

## RPN Interpretation

**RPN = Severity × Occurrence × Detection**

**Range**: 1 to 1,000

| RPN Range | Priority Level | Action Required |
|-----------|----------------|-----------------|
| 1-49 | Low | Monitor, no immediate action |
| 50-99 | Moderate | Schedule preventive maintenance |
| 100-199 | High | Perform maintenance soon |
| 200-299 | Very High | Urgent maintenance required |
| 300-1000 | Critical | Immediate action, ground aircraft |

**Example Calculations:**

1. **Oil System Failure**
   - Severity = 8 (engine damage)
   - Occurrence = 4 (occasional)
   - Detection = 3 (sensor alarm)
   - **RPN = 8 × 4 × 3 = 96** → Moderate priority

2. **Hydraulic Line Rupture**
   - Severity = 9 (loss of flight control)
   - Occurrence = 3 (rare)
   - Detection = 5 (moderate chance of early detection)
   - **RPN = 9 × 3 × 5 = 135** → High priority

3. **Landing Light Failure**
   - Severity = 3 (minor inconvenience)
   - Occurrence = 6 (happens occasionally)
   - Detection = 1 (immediately obvious)
   - **RPN = 3 × 6 × 1 = 18** → Low priority

---

## Database Implementation

### `fmea_ratings` Table

```sql
CREATE TABLE fmea_ratings (
    fmea_id INTEGER PRIMARY KEY,
    component_id INTEGER,
    failure_mode TEXT NOT NULL,
    severity INTEGER CHECK(severity BETWEEN 1 AND 10),
    occurrence INTEGER CHECK(occurrence BETWEEN 1 AND 10),
    detection INTEGER CHECK(detection BETWEEN 1 AND 10),
    rpn INTEGER,  -- Auto-calculated by trigger
    FOREIGN KEY (component_id) REFERENCES components(component_id)
);
```

### Automatic RPN Calculation (Trigger)

```sql
CREATE TRIGGER calculate_rpn
AFTER INSERT ON fmea_ratings
BEGIN
    UPDATE fmea_ratings
    SET rpn = NEW.severity * NEW.occurrence * NEW.detection
    WHERE fmea_id = NEW.fmea_id;
END;
```

**Benefit**: RPN is always consistent and automatically updated.

---

## Integration with ML Predictions

### 1. Model Predicts Failure
```python
# LSTM model predicts low RUL
predicted_rul = 25.3  # hours
confidence = 0.92
failure_mode = "Oil System Failure"
```

### 2. Link to FMEA Ratings
```sql
SELECT severity, occurrence, detection, rpn
FROM fmea_ratings
WHERE failure_mode = 'Oil System Failure'
  AND component_id = 42;
```

**Result**: Severity=8, Occurrence=4, Detection=3, RPN=96

### 3. Generate Alert
```sql
INSERT INTO alerts (component_id, alert_type, severity, rpn, message)
VALUES (42, 'PREDICTIVE_FAILURE', 'HIGH', 96,
        'Oil system failure predicted in 25 hours with 92% confidence');
```

### 4. Map to Preventive Task
```sql
SELECT pt.task_name, pt.ac_43_12c_reference, pt.pilot_allowed
FROM pdm_task_mapping m
JOIN preventive_tasks pt ON m.task_id = pt.task_id
WHERE m.failure_mode = 'Oil System Failure';
```

**Result**: "Inspect oil system (AC 43-12C Section 3.5), A&P Required"

---

## How Our System Uses FMEA

### 1. Pre-populated FMEA Ratings
The database contains FMEA ratings for common failure modes:
- Oil system failures
- Hydraulic system failures
- Electrical system failures
- Engine failures
- Landing gear failures

### 2. Dynamic Risk Assessment
When a model makes a prediction:
1. System looks up FMEA rating for that failure mode
2. Calculates RPN (via trigger)
3. Compares RPN to thresholds
4. Generates appropriate alert level

### 3. Prioritized Maintenance Queue
Dashboard displays components sorted by:
1. RPN (descending)
2. RUL (ascending)
3. Prediction confidence (descending)

---

## RPN Thresholds in Code

Defined in `config/settings.py`:

```python
# FMEA (Failure Mode and Effects Analysis) PARAMETERS
RPN_LOW_THRESHOLD: int = 50
RPN_MEDIUM_THRESHOLD: int = 100
RPN_HIGH_THRESHOLD: int = 200
RPN_CRITICAL_THRESHOLD: int = 300
```

Used in queries:

```sql
SELECT component_id, failure_mode, rpn,
       CASE
         WHEN rpn >= 300 THEN 'CRITICAL'
         WHEN rpn >= 200 THEN 'VERY_HIGH'
         WHEN rpn >= 100 THEN 'HIGH'
         WHEN rpn >= 50 THEN 'MODERATE'
         ELSE 'LOW'
       END AS priority_level
FROM fmea_ratings
ORDER BY rpn DESC;
```

---

## Example: Complete FMEA Workflow

**Scenario**: LSTM model detects oil pressure degradation.

1. **Sensor Data**:
   - Oil pressure: 28 psi (threshold: 30 psi)
   - Trend: Decreasing over last 50 flight hours

2. **ML Prediction**:
   - Model: LSTM (ID: 5)
   - Prediction: "Oil System Failure"
   - RUL: 32 hours
   - Confidence: 0.91

3. **FMEA Lookup**:
   ```sql
   SELECT * FROM fmea_ratings
   WHERE failure_mode = 'Oil System Failure'
   AND component_id = 15;
   ```
   Result: Severity=8, Occurrence=4, Detection=3, **RPN=96**

4. **Alert Classification**:
   - RPN = 96 → **Moderate** (50-99 range)
   - But confidence = 0.91 → **Elevated to High**

5. **Preventive Task Mapping**:
   ```sql
   SELECT pt.task_name, pt.description, pt.ac_43_12c_reference
   FROM pdm_task_mapping m
   JOIN preventive_tasks pt ON m.task_id = pt.task_id
   WHERE m.failure_mode = 'Oil System Failure';
   ```
   Result: "Inspect oil system for leaks, contamination, or filter blockage (AC 43-12C Section 3.5)"

6. **Dashboard Display**:
   - Component: Engine #1 (N12345)
   - Priority: HIGH
   - Action: "Inspect oil system before next flight"
   - Reference: AC 43-12C Section 3.5
   - Estimated Time: 2 hours
   - Requires: A&P Mechanic

---

## FMEA Benefits in PdM System

✅ **Quantified Risk**: Numerical RPN allows objective prioritization

✅ **Consistency**: Same failure mode always gets same RPN (if SEV/OCC/DET unchanged)

✅ **Traceability**: Every alert traces back to specific FMEA rating

✅ **Regulatory Alignment**: Links FMEA failure modes to FAA AC 43-12C tasks

✅ **Continuous Improvement**: Can update SEV/OCC/DET ratings based on fleet data

---

## Updating FMEA Ratings

As the system learns from real-world data, FMEA ratings can be adjusted:

```sql
-- Example: Oil system failures are occurring more frequently than expected
UPDATE fmea_ratings
SET occurrence = 6  -- Increased from 4
WHERE failure_mode = 'Oil System Failure';
-- RPN automatically recalculated by trigger: 8 × 6 × 3 = 144 (now HIGH priority)
```

---

## For More Information

- See [SCHEMA_OVERVIEW.md](SCHEMA_OVERVIEW.md) for database details
- See [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for ML integration
- See main [README.md](../README.md) for system overview

---

**Last Updated**: November 2025
