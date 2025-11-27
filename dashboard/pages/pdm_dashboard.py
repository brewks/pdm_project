"""
PdM Dashboard - Main Predictive Maintenance Overview.

This page provides comprehensive views of:
- Components needing attention
- Dashboard snapshot with all key metrics
- Latest model predictions
- Engine health overview

It demonstrates the complete data flow from sensor data through ML predictions
to actionable maintenance insights.

Author: General Aviation PdM Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Optional

from config.settings import (
    PAGE_LAYOUT,
    DEFAULT_REFRESH_INTERVAL,
    MAX_REFRESH_INTERVAL,
    CHART_FIGURE_SIZE,
    CHART_COLOR_PALETTE,
    LOGO_PATH
)
from config.db_utils import execute_query
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="GA PdM Dashboard", layout=PAGE_LAYOUT)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_rul_bar(df: pd.DataFrame) -> None:
    """
    Create bar chart of Remaining Useful Life predictions.

    Args:
        df: DataFrame containing RUL predictions with component_id and predicted_value
    """
    fig, ax = plt.subplots(figsize=CHART_FIGURE_SIZE)
    sns.barplot(
        data=df,
        x=df['component_id'].astype(str),
        y='predicted_value',
        ax=ax,
        palette=CHART_COLOR_PALETTE
    )
    ax.set_xlabel("Component ID")
    ax.set_ylabel("Predicted RUL (hours)")
    ax.set_title("Latest Component RUL Predictions")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def plot_confidence_rul(df: pd.DataFrame) -> None:
    """
    Create scatter plot of RUL vs Component ID with confidence sizing.

    Args:
        df: DataFrame containing predictions with confidence levels
    """
    fig, ax = plt.subplots(figsize=CHART_FIGURE_SIZE)
    sns.scatterplot(
        data=df,
        x='component_id',
        y='predicted_value',
        size='confidence',
        hue='confidence',
        palette='coolwarm',
        ax=ax,
        sizes=(50, 300)
    )
    ax.set_title("RUL vs Component ID (Size = Confidence)")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


def plot_rul_trend(df: pd.DataFrame) -> None:
    """
    Create line plot of RUL predictions over time.

    Args:
        df: DataFrame containing time-series RUL predictions
    """
    if 'prediction_time' not in df.columns:
        st.info("No time-series data available for trend plot.")
        return

    df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')

    fig, ax = plt.subplots(figsize=CHART_FIGURE_SIZE)
    sns.lineplot(
        data=df,
        x='prediction_time',
        y='predicted_value',
        hue='component_id',
        marker="o",
        ax=ax
    )
    ax.set_title("RUL Predictions Over Time")
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=60)
def load_view_data(view_name: str) -> pd.DataFrame:
    """
    Load data from a specified database view.

    Args:
        view_name: Name of the database view to query

    Returns:
        pd.DataFrame: Query results
    """
    query = f"SELECT * FROM {view_name}"
    try:
        return execute_query(query)
    except Exception as e:
        logger.error(f"Failed to load view {view_name}: {e}")
        st.error(f"Failed to load {view_name} data.")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_latest_predictions(limit: int = 100) -> pd.DataFrame:
    """
    Load latest model predictions.

    Args:
        limit: Maximum number of records to return

    Returns:
        pd.DataFrame: Latest predictions
    """
    query = f"""
        SELECT * FROM component_predictions
        ORDER BY prediction_time DESC
        LIMIT {limit}
    """
    try:
        return execute_query(query)
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        st.error("Failed to load predictions data.")
        return pd.DataFrame()


# ============================================================================
# CUSTOM STYLING
# ============================================================================

def apply_dashboard_styling() -> None:
    """Apply custom CSS styling to the dashboard."""
    st.markdown("""
    <style>
        .stApp {
            background-color: #e9edf5;
        }

        .dashboard-title {
            font-size: 28px;
            font-weight: bold;
            color: #1f2d4a;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }

        .big-font {
            font-size: 18px !important;
            color: #255D00;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(8px);
            border-right: 1px solid rgba(0,0,0,0.05);
            box-shadow: inset -3px 0 6px rgba(0,0,0,0.05);
        }

        .stSlider, .stRadio {
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.4);
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main dashboard application.

    Provides multiple view options for monitoring aircraft health and
    maintenance predictions.
    """
    logger.info("PdM Dashboard page loaded")

    apply_dashboard_styling()

    # === HEADER WITH LOGO ===
    col1, col2 = st.columns([1, 10])
    with col1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
    with col2:
        st.markdown(
            '<div class="dashboard-title">General Aviation Predictive Maintenance Dashboard</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<p class="big-font">Live aircraft system health, predictive maintenance insights, and alerts</p>',
        unsafe_allow_html=True
    )

    # === SIDEBAR CONTROLS ===
    st.sidebar.header("Dashboard Controls")

    refresh_interval = st.sidebar.slider(
        "Auto-refresh (seconds)",
        0,
        MAX_REFRESH_INTERVAL,
        DEFAULT_REFRESH_INTERVAL
    )

    view_choice = st.sidebar.radio(
        "Select View",
        [
            "Components Needing Attention",
            "Dashboard Snapshot",
            "Latest Predictions",
            "Engine Health Overview"
        ]
    )

    # === LOAD DATA BASED ON VIEW SELECTION ===
    if view_choice == "Components Needing Attention":
        df = load_view_data("components_needing_attention")
    elif view_choice == "Dashboard Snapshot":
        df = load_view_data("dashboard_snapshot_view")
    elif view_choice == "Engine Health Overview":
        df = load_view_data("engine_health_view")
    else:  # Latest Predictions
        df = load_latest_predictions(100)

    # === DISPLAY DATA ===
    st.markdown(f"""
        <h4 style="font-size:20px; font-weight:600; color:#1f2d4a; margin-bottom:6px;">
            {view_choice}
        </h4>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <p style="font-size:16px; font-weight:400; color:#1f2d4a; margin-top:0;">
            Data Summary: {len(df):,} records loaded
        </p>
    """, unsafe_allow_html=True)

    st.dataframe(df, use_container_width=True)

    if df.empty:
        st.warning("⚠ No data available for this view.")
        logger.warning(f"No data available for view: {view_choice}")
    else:
        # === VISUALIZATIONS FOR PREDICTIONS VIEW ===
        if view_choice == "Latest Predictions":
            st.write("### Predicted Remaining Useful Life (RUL)")
            plot_rul_bar(df)

            st.write("### RUL vs Component ID with Confidence")
            plot_confidence_rul(df)

            st.write("### RUL Prediction Trends")
            plot_rul_trend(df)

        # === CRITICAL ALERTS ===
        if "confidence" in df.columns and "prediction_type" in df.columns:
            critical_alerts = df[
                (df["confidence"] > 0.9) &
                (df["prediction_type"] == "failure")
            ]
            if not critical_alerts.empty:
                st.error(f"🚨 {len(critical_alerts)} CRITICAL failure predictions detected!")
                logger.warning(f"Critical alerts detected: {len(critical_alerts)}")

        # === CONFIDENCE FILTER ===
        if "confidence" in df.columns:
            st.markdown("---")
            st.subheader("Filter by Confidence Level")

            conf_level = st.slider("Minimum Confidence", 0.0, 1.0, 0.7, 0.05)
            filtered_df = df[df['confidence'] >= conf_level]

            st.write(f"### Filtered Predictions (Confidence ≥ {conf_level:.0%})")
            st.dataframe(filtered_df, use_container_width=True)

            if not filtered_df.empty:
                st.download_button(
                    label="📥 Download Filtered Data",
                    data=filtered_df.to_csv(index=False).encode(),
                    file_name=f"filtered_predictions_conf{conf_level:.0%}.csv",
                    mime="text/csv"
                )

    # === AUTO-REFRESH ===
    if refresh_interval > 0:
        st.info(f"⏳ Auto-refreshing every {refresh_interval} seconds...")
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
