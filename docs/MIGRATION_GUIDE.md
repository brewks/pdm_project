# Migration Guide: Original → Sanitized Codebase

This document explains the changes made during the sanitization and standardization process, and how to migrate from the original codebase to the sanitized version.

---

## Summary of Changes

### 1. **Folder Restructuring**

**Before:**
```
ga_maintenance/
├── app.py
├── home.py
├── utils.py
├── generate_degrading_sensor_data.py
├── pages/
├── scripts/
├── full_pdm_seed.sql
├── insert_*.sql
└── requirements.txt
```

**After:**
```
ga_maintenance/
├── config/                      # NEW: Centralized configuration
│   ├── settings.py
│   ├── db_utils.py
│   └── logging_config.py
├── data_generation/             # NEW: Organized data generation
│   └── synthetic_sensor_generator.py
├── ml/                          # NEW: Machine learning modules
│   ├── models/
│   └── train_models.py
├── dashboard/                   # REORGANIZED: Dashboard files
│   ├── app.py
│   └── pages/
├── database/                    # NEW: SQL files and database
│   ├── full_pdm_seed.sql
│   └── insert_*.sql
├── docs/                        # NEW: Comprehensive documentation
│   ├── SCHEMA_OVERVIEW.md
│   ├── FMEA_DOCUMENTATION.md
│   ├── PIPELINE_OVERVIEW.md
│   └── MIGRATION_GUIDE.md
├── logs/                        # NEW: Structured logging
└── backups/                     # NEW: Database backups
```

---

## Key Improvements

### 1. Configuration Management

**Before:** Hard-coded paths scattered throughout files
```python
# Original code
DB_PATH = "C:/Users/workd/Desktop/ga_maintenance/PdM/ga_maintenance.db"
```

**After:** Centralized in `config/settings.py`
```python
# Sanitized code
from config.settings import DATABASE_PATH
# DATABASE_PATH automatically computed relative to project root
```

**Migration Steps:**
1. Remove all hard-coded paths from your code
2. Import from `config.settings` instead
3. All paths now work regardless of installation location

---

### 2. Database Access

**Before:** Duplicate `load_df()` and `validate_metrics()` functions in multiple files
```python
# Duplicated in app.py, home.py, pages/*.py
def load_df(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
```

**After:** Single source in `config/db_utils.py`
```python
# Used everywhere
from config.db_utils import execute_query

df = execute_query("SELECT * FROM components")
```

**Migration Steps:**
1. Replace all `load_df(query)` calls with `execute_query(query)`
2. Remove duplicate function definitions
3. Import from `config.db_utils`

---

### 3. Logging

**Before:** Print statements for debugging
```python
# Original code
print(f"✅ Inserted {count} records")
```

**After:** Structured logging
```python
# Sanitized code
from config.logging_config import setup_logger

logger = setup_logger(__name__)
logger.info(f"Inserted {count:,} records")
```

**Benefits:**
- All logs written to `logs/pdm_system.log`
- Timestamps, log levels, module names included
- Can filter by severity (INFO, WARNING, ERROR)

**Migration Steps:**
1. Replace `print()` statements with `logger.info()`, `logger.warning()`, `logger.error()`
2. Add logger setup at module top: `logger = setup_logger(__name__)`

---

### 4. Type Hints and Docstrings

**Before:** No type information
```python
# Original code
def generate_degrading_sensor_data(db_path, top_params, num_components=10, num_records=1000):
    # ... implementation
```

**After:** Full type hints and comprehensive docstrings
```python
# Sanitized code
def generate_synthetic_data(
    db_path: Path,
    params: Optional[List[str]] = None,
    num_components: int = DEFAULT_NUM_COMPONENTS,
    num_records: int = DEFAULT_NUM_RECORDS
) -> None:
    """
    Generate synthetic sensor data with realistic degradation patterns.

    Args:
        db_path: Path to the SQLite database
        params: List of sensor parameters to generate
        num_components: Number of components to generate data for
        num_records: Number of records per component per parameter

    Example:
        >>> from pathlib import Path
        >>> db = Path("database/ga_maintenance.db")
        >>> generate_synthetic_data(db, num_components=20, num_records=2000)
    """
    # ... implementation
```

**Benefits:**
- IDE autocomplete and type checking
- Clear function contracts
- Usage examples in docstrings

---

### 5. Code Organization

**Before:** All logic in a single file/function

**After:** Modular, single-responsibility functions

**Example: ML Training**

**Before (conceptual):**
```python
# All in one giant function
def train_models():
    # Extract data
    # Clean data
    # Transform data
    # Train RF
    # Train LSTM
    # Evaluate
    # Log predictions
    # Plot results
```

**After:**
```python
# Separated into focused functions
def extract_sensor_data() -> pd.DataFrame: ...
def clean_sensor_data(df: pd.DataFrame) -> pd.DataFrame: ...
def pivot_sensor_data(df: pd.DataFrame) -> pd.DataFrame: ...
def normalize_features(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, MinMaxScaler]: ...
def create_sequences(...) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def train_random_forest(...) -> Tuple[RandomForestRegressor, Dict[str, float]]: ...
def train_lstm(...) -> Tuple[Sequential, Dict[str, float]]: ...
def log_predictions_to_database(...) -> None: ...
def plot_predictions(...) -> None: ...

def main():
    # Orchestrates the pipeline
    sensor_df = extract_sensor_data()
    sensor_df = clean_sensor_data(sensor_df)
    # ... etc.
```

**Benefits:**
- Easier to test individual functions
- Clearer data flow
- Reusable components

---

### 6. Dashboard Files

**Before:** `app.py` and `home.py` with duplicate code and styling

**After:** Clean separation
- `dashboard/app.py` - Main component health dashboard
- `dashboard/pages/pdm_dashboard.py` - Predictions and analytics views
- `dashboard/pages/model_monitor.py` - Model performance tracking
- `dashboard/pages/due_preventive_tasks.py` - FAA-aligned task list

**Migration Steps:**
1. Use `streamlit run dashboard/app.py` for main dashboard
2. Use `streamlit run dashboard/pages/pdm_dashboard.py` for PdM views
3. All pages now use centralized `config.db_utils` for data loading

---

### 7. Constants and Magic Numbers

**Before:** Magic numbers scattered in code
```python
# Original code
lstm_units = 64
epochs = 20
batch_size = 16
confidence = 0.85
```

**After:** Named constants in `config/settings.py`
```python
# In config/settings.py
LSTM_UNITS: int = 64
LSTM_EPOCHS: int = 20
LSTM_BATCH_SIZE: int = 16
DEFAULT_CONFIDENCE: float = 0.85

# In your code
from config.settings import LSTM_UNITS, LSTM_EPOCHS, LSTM_BATCH_SIZE

model = Sequential([
    Input(shape=(LSTM_SEQUENCE_LENGTH, len(TOP_SENSOR_PARAMS))),
    LSTM(LSTM_UNITS),
    Dense(1)
])
model.compile(optimizer=Adam(LSTM_LEARNING_RATE), loss='mse')
model.fit(x_train, y_train, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, ...)
```

**Benefits:**
- Single source of truth
- Easy to adjust hyperparameters
- Self-documenting code

---

## File Mapping

| Original File | New Location | Notes |
|--------------|--------------|-------|
| `app.py` | `dashboard/app.py` | Sanitized, uses config module |
| `home.py` | REMOVED | Functionality merged into `dashboard/app.py` |
| `utils.py` | `config/db_utils.py` | Expanded with type hints, logging |
| `generate_degrading_sensor_data.py` | `data_generation/synthetic_sensor_generator.py` | Class-based, documented |
| `pages/*.py` | `dashboard/pages/*.py` | All sanitized with consistent style |
| `full_pdm_seed.sql` | `database/full_pdm_seed.sql` | Moved, ready for documentation |
| `insert_*.sql` | `database/insert_*.sql` | Organized in database folder |
| `requirements.txt` | `requirements.txt` | Updated with version constraints |

---

## Backward Compatibility

The sanitized codebase is **not** backward compatible with the original due to:

1. **Import changes**: Must import from new module paths
2. **Function renames**: `load_df()` → `execute_query()`
3. **Path changes**: All paths now computed via `config.settings`

**However**, migration is straightforward:
1. Update imports
2. Replace function calls
3. Remove hard-coded paths

---

## Quick Migration Checklist

If you have custom code that depends on the original structure:

- [ ] Update imports: `from config.db_utils import execute_query`
- [ ] Replace `DB_PATH` with `from config.settings import DATABASE_PATH`
- [ ] Replace `load_df(query)` with `execute_query(query)`
- [ ] Replace `print()` with `logger.info()`
- [ ] Move any custom scripts to appropriate folders (data_generation/, ml/, etc.)
- [ ] Update dashboard launch commands to new paths

---

## What Stayed the Same

**Database schema, logic, and behavior are 100% preserved:**

- ✅ All table names, column names, constraints unchanged
- ✅ All triggers, views, indices unchanged
- ✅ All FMEA/RPN logic unchanged
- ✅ All FAA AC 43-12C mappings unchanged
- ✅ All ML model architectures and hyperparameters unchanged (by default)
- ✅ All prediction logic unchanged
- ✅ All dashboard visualizations unchanged

**Only improvements:**
- Code organization
- Documentation
- Type safety
- Logging
- Configurability

---

## Testing Your Migration

```bash
# 1. Verify configuration
python -c "from config.settings import validate_paths; print('Paths valid:', validate_paths())"

# 2. Test database connection
python -c "from config.db_utils import check_database_exists; print('DB exists:', check_database_exists())"

# 3. Generate synthetic data (optional)
python data_generation/synthetic_sensor_generator.py

# 4. Train models (optional, requires data)
python ml/train_models.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

Expected result: Dashboard loads without errors, data displays correctly.

---

## Getting Help

If you encounter issues during migration:

1. **Check logs**: `logs/pdm_system.log` contains detailed error messages
2. **Verify paths**: Ensure `database/ga_maintenance.db` exists
3. **Check imports**: Make sure `config/`, `ml/`, etc. are in your Python path
4. **Review README**: See main [README.md](../README.md) for full documentation

---

## Summary

The sanitized codebase provides:

✅ **Better structure** - Modular, organized folders
✅ **Centralized config** - No hard-coded paths
✅ **Proper logging** - Structured, filterable logs
✅ **Type safety** - Full type hints
✅ **Documentation** - Comprehensive docstrings and READMEs
✅ **Best practices** - Follows industry standards
✅ **Interview-ready** - Aligns with Airbus evaluation framework

While maintaining:

✅ **Same functionality** - All features work identically
✅ **Same database** - Schema and logic unchanged
✅ **Same models** - ML architecture preserved
✅ **Same dashboards** - Visual appearance and features unchanged

---

**Migration Status**: Complete ✅
**Version**: 1.0.0
**Date**: November 2025
