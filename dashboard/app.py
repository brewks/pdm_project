"""
Main Streamlit Application Entry Point for GA PdM Dashboard.

This is the primary entry point for the General Aviation Predictive Maintenance
Dashboard. It provides a multi-page interface for monitoring aircraft health,
viewing model predictions, and tracking preventive maintenance tasks.

The dashboard demonstrates the complete Model Integration Pipeline:
Data → ETL → ML Models → Predictions → Dashboard Visualization

Author: Ndubuisi Chibuogwu
Date: Dec 2024- July 2025
"""

import streamlit as st
import pandas as pd
import json
from typing import Optional
import altair as alt

from config.settings import (
    DATABASE_PATH,
    SEED_SQL_PATH,
    PAGE_TITLE,
    PAGE_LAYOUT,
    DARK_MODE_BG,
    DARK_MODE_CARD_BG,
    DARK_MODE_TEXT,
    DARK_MODE_METRIC_BG,
    LIGHT_MODE_BG,
    LIGHT_MODE_CARD_BG,
    LIGHT_MODE_TEXT,
    LIGHT_MODE_METRIC_BG,
    LOGO_PATH
)
from config.db_utils import (
    execute_query,
    check_database_exists,
    restore_database_from_seed,
    validate_metrics_json,
    parse_metrics_json
)
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def initialize_database() -> bool:
    """
    Check if database exists and restore from seed if missing.

    Returns:
        bool: True if database is available, False otherwise
    """
    if not check_database_exists():
        st.warning("Database file not found. Attempting to restore from SQL seed...")
        logger.warning("Database not found. Initiating restoration.")

        if restore_database_from_seed(SEED_SQL_PATH):
            st.success("Database successfully restored from seed file.")
            logger.info("Database restored successfully.")
            return True
        else:
            st.error("Database restoration failed. Please check logs.")
            logger.error("Database restoration failed.")
            return False

    return True


# ============================================================================
# STYLING AND THEME
# ============================================================================

def apply_custom_styling(dark_mode: bool = False) -> None:
    """
    Apply custom CSS styling to the Streamlit dashboard.

    Args:
        dark_mode: If True, applies dark mode styling
    """
    if dark_mode:
        background = DARK_MODE_BG
        card_bg = DARK_MODE_CARD_BG
        text_color = DARK_MODE_TEXT
        metric_bg = DARK_MODE_METRIC_BG
        button_bg = "#2980b9"
    else:
        background = LIGHT_MODE_BG
        card_bg = LIGHT_MODE_CARD_BG
        text_color = LIGHT_MODE_TEXT
        metric_bg = LIGHT_MODE_METRIC_BG
        button_bg = "#1565c0"

    st.markdown(f"""
    <style>
    .stApp {{
        background: {background};
        font-family: 'Segoe UI', sans-serif;
        color: {text_color};
    }}

    .header-bar {{
        background: {metric_bg};
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 26px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    }}

    .card {{
        background: {card_bg};
        color: {text_color};
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        margin-bottom: 15px;
    }}

    .metric-card {{
        background: {metric_bg};
        padding: 12px;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }}

    .stButton > button {{
        background-color: {button_bg};
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 6px 12px;
    }}

    .stButton > button:hover {{
        background-color: #004d99;
    }}

    textarea.stTextArea textarea {{
        background-color: {"#2c3e50" if dark_mode else "#ffffff"};
        color: {"#f1f1f1" if dark_mode else "#333333"};
        border: 1px solid #888;
        border-radius: 5px;
        font-size: 14px;
    }}
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=60)
def load_components_data() -> pd.DataFrame:
    """
    Load components data from database with caching.

    Returns:
        pd.DataFrame: Components data
    """
    query = """
        SELECT component_id, tail_number, name, condition,
               remaining_useful_life, last_health_score
        FROM components
    """
    try:
        return execute_query(query)
    except Exception as e:
        logger.error(f"Failed to load components data: {e}")
        st.error("Failed to load components data.")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_predictions_data() -> pd.DataFrame:
    """
    Load component predictions from database with caching.

    Returns:
        pd.DataFrame: Predictions data
    """
    query = """
        SELECT * FROM component_predictions
        ORDER BY prediction_time DESC
    """
    try:
        return execute_query(query)
    except Exception as e:
        logger.error(f"Failed to load predictions data: {e}")
        st.error("Failed to load predictions data.")
        return pd.DataFrame()


# ============================================================================
# COMPONENT DETAIL VIEW
# ============================================================================

def display_component_details(
    comp_data: pd.Series,
    comp_preds: pd.DataFrame,
    selected_name: str
) -> None:
    """
    Display detailed information for a selected component.

    Args:
        comp_data: Series containing component data
        comp_preds: DataFrame containing component predictions
        selected_name: Display name of the selected component
    """
    # Left column: Component info and alerts
    col1, col2 = st.columns([2, 1])

    with col1:
        # Component overview card
        st.markdown(f"""
        <div class="card" style="background:#3b5998; color:white;">
        <h4 style="color:white; font-weight:700; margin-bottom:10px;">{selected_name}</h4>
        <b>Condition:</b> {comp_data['condition']}<br>
        <b>Remaining Useful Life:</b> {comp_data['remaining_useful_life']:.2f} hours<br>
        <b>Health Score:</b> {comp_data['last_health_score']}
        </div>
        """, unsafe_allow_html=True)

        # Critical alerts
        critical_preds = comp_preds[
            (comp_preds['prediction_type'] == 'failure') &
            (comp_preds['confidence'] > 0.9)
        ]

        if not critical_preds.empty:
            row = critical_preds.iloc[0]
            st.markdown(f"""
            <div class="card" style="background:#e53935; color:white;">
            <b>⚠ Critical Alert</b><br>
            Predicted Failure: {row['time_horizon']}<br>
            Confidence: {row['confidence']*100:.1f}%<br>
            Explanation: {row['explanation']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="background:#43a047; color:white;">
            <b>✓ No Critical Alerts</b>
            </div>
            """, unsafe_allow_html=True)

        # RUL trend chart (Altair)
        if not comp_preds.empty:
            rul_data = comp_preds[comp_preds['prediction_type'] == 'remaining_life']
            if not rul_data.empty:
                rul_data['prediction_time'] = pd.to_datetime(rul_data['prediction_time'])
                chart = alt.Chart(rul_data).mark_line(point=True).encode(
                    x=alt.X('prediction_time:T', title='Prediction Time'),
                    y=alt.Y('predicted_value:Q', title='Remaining Useful Life (hrs)'),
                    tooltip=['prediction_time:T', 'predicted_value', 'confidence']
                ).properties(
                    title="Remaining Useful Life Over Time",
                    width=600,
                    height=300
                )
                st.altair_chart(chart, use_container_width=True)

    with col2:
        # Metrics summary
        avg_conf = comp_preds['confidence'].mean() * 100 if not comp_preds.empty else 0
        crit_count = len(critical_preds)
        avg_rul = comp_data['remaining_useful_life']

        st.markdown(f"""
        <div class="metric-card">
        <b>Avg Confidence</b><br>
        <span style="font-size:22px;">{avg_conf:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="background:#c62828;">
        <b>Critical Failures</b><br>
        <span style="font-size:22px;">{crit_count}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="background:#1565c0;">
        <b>Remaining Useful Life</b><br>
        <span style="font-size:22px;">{avg_rul:.2f}h</span>
        </div>
        """, unsafe_allow_html=True)

        # Model performance metrics
        if not comp_preds.empty:
            model_id = comp_preds['model_id'].iloc[0]
            metrics_query = f"""
                SELECT performance_metrics FROM predictive_models
                WHERE model_id = {model_id}
            """
            try:
                metrics_df = execute_query(metrics_query)
                if not metrics_df.empty and metrics_df['performance_metrics'].iloc[0]:
                    metrics_json = metrics_df['performance_metrics'].iloc[0]
                    metrics = parse_metrics_json(metrics_json)

                    if metrics:
                        st.markdown(f"""
                        <div class="card" style="background:#00796b; color:white;">
                        <b>Model Performance</b><br>
                        Precision: {metrics['precision']*100:.1f}% |
                        Recall: {metrics['recall']*100:.1f}%<br>
                        Accuracy: {metrics['accuracy']*100:.1f}% |
                        F1: {metrics['f1_score']*100:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠ Invalid performance metrics")
            except Exception as e:
                logger.error(f"Failed to load model metrics: {e}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.

    Orchestrates the dashboard layout, database initialization, data loading,
    and component visualization.
    """
    logger.info("Dashboard application started")

    # Initialize database
    if not initialize_database():
        st.stop()

    # Sidebar: Dark mode toggle
    dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
    apply_custom_styling(dark_mode)

    # Header
    st.markdown(
        '<div class="header-bar">General Aviation Predictive Maintenance Dashboard</div>',
        unsafe_allow_html=True
    )

    # Load data
    components_df = load_components_data()
    predictions_df = load_predictions_data()

    if components_df.empty:
        st.warning("⚠ No aircraft components available. Database may not have loaded correctly.")
        logger.warning("No components data available")
        return

    # Component selector
    component_names = [
        f"{row['tail_number']} - {row['name']}"
        for _, row in components_df.iterrows()
    ]
    component_map = {
        f"{row['tail_number']} - {row['name']}": row['component_id']
        for _, row in components_df.iterrows()
    }

    selected_component = st.selectbox("Select Aircraft Component:", component_names)
    comp_id = component_map[selected_component]

    comp_data = components_df[components_df['component_id'] == comp_id].iloc[0]
    comp_preds = predictions_df[predictions_df['component_id'] == comp_id]

    # Display component details
    display_component_details(comp_data, comp_preds, selected_component)

    # JSON validator section
    st.markdown("---")
    st.markdown("""
    <div class="card">
    <h4 style="margin-bottom:10px;">🧪 Test Performance Metrics JSON</h4>
    </div>
    """, unsafe_allow_html=True)

    input_metrics = st.text_area("Enter Performance Metrics JSON:", height=150)

    if st.button("Validate JSON"):
        if validate_metrics_json(input_metrics):
            st.success("✅ Valid performance metrics JSON!")
            logger.info("User validated metrics JSON successfully")
        else:
            st.error("❌ Invalid or missing required fields (precision, recall, accuracy, f1_score).")
            logger.warning("User submitted invalid metrics JSON")


if __name__ == "__main__":
    main()
