"""
Database utility functions for the GA Predictive Maintenance System.

This module provides centralized database connection and query utilities
used across all components of the PdM system.

Author: Ndubuisi Chibuogwu
Date: 2025
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd

from config.settings import DATABASE_PATH, REQUIRED_METRIC_FIELDS
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Create and return a database connection.

    Args:
        db_path: Path to the database file. If None, uses default from settings.

    Returns:
        sqlite3.Connection: Database connection object

    Raises:
        sqlite3.Error: If connection fails
    """
    path = db_path or DATABASE_PATH
    try:
        conn = sqlite3.connect(str(path))
        logger.debug(f"Database connection established to {path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database at {path}: {e}")
        raise


def execute_query(
    query: str,
    params: Optional[tuple] = None,
    db_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a pandas DataFrame.

    This is the centralized function for all read queries in the system.
    It handles connection management and error logging automatically.

    Args:
        query: SQL query string to execute
        params: Optional tuple of parameters for parameterized queries
        db_path: Optional database path. Uses default if not provided.

    Returns:
        pd.DataFrame: Query results as a DataFrame

    Raises:
        sqlite3.Error: If query execution fails
        pd.io.sql.DatabaseError: If DataFrame conversion fails

    Example:
        >>> df = execute_query("SELECT * FROM components WHERE component_id = ?", (42,))
    """
    try:
        with get_connection(db_path) as conn:
            if params:
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn)
            logger.debug(f"Query executed successfully. Returned {len(df)} rows.")
            return df
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        logger.error(f"Query execution failed: {e}")
        logger.error(f"Query: {query}")
        raise


def execute_write(
    query: str,
    params: Optional[tuple] = None,
    db_path: Optional[Path] = None
) -> int:
    """
    Execute a write SQL statement (INSERT, UPDATE, DELETE).

    Args:
        query: SQL statement to execute
        params: Optional tuple of parameters for parameterized queries
        db_path: Optional database path

    Returns:
        int: Number of rows affected

    Raises:
        sqlite3.Error: If execution fails
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            rows_affected = cursor.rowcount
            logger.debug(f"Write query executed. {rows_affected} rows affected.")
            return rows_affected
    except sqlite3.Error as e:
        logger.error(f"Write query execution failed: {e}")
        logger.error(f"Query: {query}")
        raise


def execute_many(
    query: str,
    param_list: List[tuple],
    db_path: Optional[Path] = None
) -> int:
    """
    Execute a parameterized query multiple times (batch insert/update).

    Args:
        query: SQL statement with placeholders
        param_list: List of parameter tuples
        db_path: Optional database path

    Returns:
        int: Number of rows affected

    Raises:
        sqlite3.Error: If execution fails
    """
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(query, param_list)
            conn.commit()
            rows_affected = cursor.rowcount
            logger.debug(f"Batch query executed. {rows_affected} rows affected.")
            return rows_affected
    except sqlite3.Error as e:
        logger.error(f"Batch query execution failed: {e}")
        raise


def execute_script(
    script_path: Path,
    db_path: Optional[Path] = None
) -> None:
    """
    Execute a SQL script file (e.g., schema initialization, seed data).

    Args:
        script_path: Path to the SQL script file
        db_path: Optional database path

    Raises:
        FileNotFoundError: If script file doesn't exist
        sqlite3.Error: If script execution fails
    """
    if not script_path.exists():
        raise FileNotFoundError(f"SQL script not found: {script_path}")

    try:
        with get_connection(db_path) as conn:
            with open(script_path, 'r', encoding='utf-8') as f:
                sql_script = f.read()
            conn.executescript(sql_script)
            logger.info(f"SQL script executed successfully: {script_path}")
    except sqlite3.Error as e:
        logger.error(f"Failed to execute SQL script {script_path}: {e}")
        raise


def validate_metrics_json(metrics_json: str) -> bool:
    """
    Validate that a JSON string contains all required performance metric fields.

    This function ensures that model performance metrics stored in the database
    conform to the expected schema with precision, recall, accuracy, and f1_score.

    Args:
        metrics_json: JSON string to validate

    Returns:
        bool: True if valid, False otherwise

    Example:
        >>> json_str = '{"precision": 0.95, "recall": 0.92, "accuracy": 0.94, "f1_score": 0.93}'
        >>> validate_metrics_json(json_str)
        True
    """
    try:
        data = json.loads(metrics_json)
        is_valid = all(field in data for field in REQUIRED_METRIC_FIELDS)

        if not is_valid:
            missing = [f for f in REQUIRED_METRIC_FIELDS if f not in data]
            logger.warning(f"Invalid metrics JSON. Missing fields: {missing}")

        return is_valid

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Invalid JSON format in metrics: {e}")
        return False


def parse_metrics_json(metrics_json: str) -> Optional[Dict[str, float]]:
    """
    Parse and validate model performance metrics JSON.

    Args:
        metrics_json: JSON string containing metrics

    Returns:
        Dict[str, float]: Parsed metrics dictionary, or None if invalid

    Example:
        >>> metrics = parse_metrics_json('{"precision": 0.95, ...}')
        >>> metrics['precision']
        0.95
    """
    if not validate_metrics_json(metrics_json):
        return None

    try:
        return json.loads(metrics_json)
    except json.JSONDecodeError:
        return None


def check_database_exists(db_path: Optional[Path] = None) -> bool:
    """
    Check if the database file exists.

    Args:
        db_path: Optional database path. Uses default if not provided.

    Returns:
        bool: True if database exists, False otherwise
    """
    path = db_path or DATABASE_PATH
    exists = path.exists()

    if not exists:
        logger.warning(f"Database file not found at {path}")

    return exists


def restore_database_from_seed(
    seed_path: Path,
    db_path: Optional[Path] = None
) -> bool:
    """
    Restore database from a SQL seed file.

    Args:
        seed_path: Path to the SQL seed file
        db_path: Optional database path

    Returns:
        bool: True if restoration succeeded, False otherwise
    """
    try:
        execute_script(seed_path, db_path)
        logger.info("Database successfully restored from seed file")
        return True
    except Exception as e:
        logger.error(f"Database restoration failed: {e}")
        return False


def get_table_names(db_path: Optional[Path] = None) -> List[str]:
    """
    Get list of all table names in the database.

    Args:
        db_path: Optional database path

    Returns:
        List[str]: List of table names
    """
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    df = execute_query(query, db_path=db_path)
    return df['name'].tolist()


def get_table_row_count(table_name: str, db_path: Optional[Path] = None) -> int:
    """
    Get the number of rows in a table.

    Args:
        table_name: Name of the table
        db_path: Optional database path

    Returns:
        int: Number of rows in the table
    """
    query = f"SELECT COUNT(*) as count FROM {table_name};"
    df = execute_query(query, db_path=db_path)
    return int(df['count'].iloc[0])
