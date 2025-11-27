"""
Model Monitoring Dashboard Page.

This page provides visibility into ML model performance metrics, allowing users to:
- View all available predictive models
- Inspect detailed performance metrics (precision, recall, accuracy, F1)
- Download model metrics
- Validate custom metrics JSON

This supports the "Model Integration Pipeline" bucket in the Airbus evaluation framework.

Author: General Aviation PdM Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import json
from typing import Optional, Dict

from config.db_utils import execute_query, validate_metrics_json, parse_metrics_json
from config.logging_config import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(page_title="Model Monitoring", layout="wide")


# ============================================================================
# STYLING
# ============================================================================

def apply_styling(dark_mode: bool = False) -> Dict[str, str]:
    """
    Apply custom styling and return color scheme.

    Args:
        dark_mode: If True, use dark mode colors

    Returns:
        Dict[str, str]: Color scheme dictionary
    """
    if dark_mode:
        colors = {
            "background": "linear-gradient(135deg, #121212, #2c3e50)",
            "text": "#f1f1f1",
            "card": "#1e272e"
        }
    else:
        colors = {
            "background": "linear-gradient(135deg, #e8f0f8, #ffffff)",
            "text": "#333333",
            "card": "#ffffff"
        }

    st.markdown(f"""
    <style>
    .stApp {{
        background: {colors['background']};
        color: {colors['text']};
        font-family: 'Segoe UI', sans-serif;
    }}
    .card {{
        background: {colors['card']};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

    return colors


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data(ttl=300)
def load_models() -> pd.DataFrame:
    """
    Load all predictive models from the database.

    Returns:
        pd.DataFrame: Models with ID, name, algorithm, and metrics
    """
    query = """
        SELECT model_id, model_name, model_type AS algorithm, performance_metrics
        FROM predictive_models
        ORDER BY model_id DESC
    """
    try:
        return execute_query(query)
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        st.error("Failed to load model data.")
        return pd.DataFrame()


# ============================================================================
# METRIC DISPLAY
# ============================================================================

def display_model_metrics(model_id: int, metrics_json: str) -> None:
    """
    Display model performance metrics in a formatted layout.

    Args:
        model_id: ID of the model
        metrics_json: JSON string containing performance metrics
    """
    st.subheader(f"Performance Metrics for Model ID {model_id}")

    metrics = parse_metrics_json(metrics_json)

    if metrics:
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
        col2.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        col3.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.1f}%")

        # Provide download button
        st.download_button(
            label="📥 Download Metrics as JSON",
            data=json.dumps(metrics, indent=2),
            file_name=f"model_{model_id}_metrics.json",
            mime="application/json"
        )

        logger.info(f"Displayed metrics for model {model_id}")

    else:
        st.error("❌ Invalid or missing metric fields in database JSON.")
        logger.error(f"Invalid metrics JSON for model {model_id}")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main model monitoring application.

    Displays all models and their performance metrics, supporting
    model traceability and quality assurance.
    """
    logger.info("Model Monitoring page loaded")

    # Sidebar: Dark mode toggle
    dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
    apply_styling(dark_mode)

    # Header
    st.markdown(
        "<div class='card'><h2>📊 Predictive Model Monitoring Dashboard</h2></div>",
        unsafe_allow_html=True
    )

    # Load models
    model_df = load_models()

    if model_df.empty:
        st.warning("⚠ No models found in database.")
        logger.warning("No models available in database")
        return

    # Display model table
    st.subheader("Available Models")
    st.dataframe(
        model_df[["model_id", "model_name", "algorithm"]],
        use_container_width=True
    )

    # Model selection for detailed metrics
    selected_model = st.selectbox(
        "Select Model to View Detailed Metrics",
        model_df["model_id"].tolist()
    )

    metrics_json = model_df[model_df["model_id"] == selected_model]["performance_metrics"].values[0]

    # Display metrics
    display_model_metrics(selected_model, metrics_json)

    # JSON Validator Section
    st.markdown("---")
    st.markdown(
        "<div class='card'><h4>🧪 Test Your Own Metrics JSON</h4></div>",
        unsafe_allow_html=True
    )

    user_json = st.text_area("Paste JSON:", height=150, help="Enter a JSON object with precision, recall, accuracy, and f1_score")

    if st.button("Validate JSON"):
        if validate_metrics_json(user_json):
            st.success("✅ Valid performance metrics JSON!")
            logger.info("User validated custom metrics JSON successfully")

            # Display parsed values
            parsed = parse_metrics_json(user_json)
            if parsed:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Precision", f"{parsed['precision']:.3f}")
                col2.metric("Recall", f"{parsed['recall']:.3f}")
                col3.metric("Accuracy", f"{parsed['accuracy']:.3f}")
                col4.metric("F1 Score", f"{parsed['f1_score']:.3f}")
        else:
            st.error("❌ Invalid or missing required fields (precision, recall, accuracy, f1_score).")
            logger.warning("User submitted invalid metrics JSON")


if __name__ == "__main__":
    main()
