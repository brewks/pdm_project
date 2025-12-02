import streamlit as st
import pandas as pd
import json
from utils import load_df, validate_metrics

# Configure the page layout and title
st.set_page_config(page_title="Model Monitoring", layout="wide")

# === DARK MODE TOGGLE ===
# Simple switch in the sidebar to flip between light and dark themes
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")

if dark_mode:
    # Colors for dark theme
    background = "linear-gradient(135deg, #0f0f0f, #1f2937)"
    text_color = "#f5f7fa"
    card_color = "rgba(255, 255, 255, 0.05)"
    border_color = "rgba(255, 255, 255, 0.15)"
else:
    # Colors for light theme
    background = "linear-gradient(135deg, #e8f0f8, #ffffff)"
    text_color = "#1a1a1a"
    card_color = "#ffffff"
    border_color = "rgba(0, 0, 0, 0.1)"

# === GLOBAL STYLING ===
# Inject a bit of custom CSS to make the dashboard look more like a real app
st.markdown(f"""
<style>
/* App background and base font */
.stApp {{
    background: {background};
    color: {text_color};
    font-family: 'Segoe UI', sans-serif;
}}

/* Generic card container used across the page */
.card {{
    background: {card_color};
    padding: 20px;
    border-radius: 16px;
    border: 1px solid {border_color};
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    margin-bottom: 25px;
    transition: transform 0.2s ease-in-out;
}}
.card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}}

/* Headings for sections */
.section-title {{
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
}}

/* Subtle hover effect on table rows */
.dataframe tbody tr:hover {{
    background: rgba(150,150,150,0.1);
}}

/* Style around metric widgets */
.metric {{
    background: {card_color};
    padding: 12px !important;
    border-radius: 12px !important;
    border: 1px solid {border_color};
    box-shadow: 0px 4px 15px rgba(0,0,0,0.25);
}}
</style>
""", unsafe_allow_html=True)

# === TITLE CARD ===
# Simple title block at the top of the page
st.markdown(
    "<div class='card'><div class='section-title'>📊 Predictive Model Monitoring Dashboard</div></div>",
    unsafe_allow_html=True
)

# === LOAD DATA ===
# Pull the list of models and their metrics from the predictive_models table
model_df = load_df("""
    SELECT model_id, model_name, model_type AS algorithm, performance_metrics 
    FROM predictive_models 
    ORDER BY model_id DESC
""")

# === DISPLAY TABLE ===
# Show a compact table of the models available for selection
st.markdown("<div class='section-title'>Available Models</div>", unsafe_allow_html=True)
st.dataframe(model_df[["model_id", "model_name", "algorithm"]])

# === METRIC EXPLORATION PANEL ===
# Let the user pick which model to inspect
selected_model = st.selectbox("Select Model to View Metrics", model_df["model_id"])
metrics_json = model_df[model_df["model_id"] == selected_model]["performance_metrics"].values[0]

st.markdown(
    f"<div class='section-title'>Performance Metrics for Model ID {selected_model}</div>",
    unsafe_allow_html=True
)

# If the JSON stored in the DB is valid, unpack and show the metrics
if validate_metrics(metrics_json):
    metrics = json.loads(metrics_json)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    col2.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    col3.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.1f}%")

    # Quick download option so someone can grab the metrics as a JSON file
    st.download_button(
        label="📥 Download Metrics JSON",
        data=json.dumps(metrics, indent=2),
        file_name=f"model_{selected_model}_metrics.json",
        mime="application/json"
    )
else:
    st.error("Invalid or missing metric fields.")

# === JSON VALIDATOR ===
# Small helper section where you can paste raw JSON to check the format
st.markdown("---")
st.markdown("<div class='card'><h4>🧪 Validate Your Metrics JSON</h4></div>", unsafe_allow_html=True)

user_json = st.text_area("Paste JSON:", height=150)

if st.button("Validate JSON"):
    if validate_metrics(user_json):
        st.success("✅ Valid performance metrics JSON!")
    else:
        st.error("❌ Missing required fields: precision, recall, accuracy, f1_score.")
