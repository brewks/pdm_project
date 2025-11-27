import streamlit as st
import json
from utils import load_df, validate_metrics

# === DARK MODE ===
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")

if dark_mode:
    card_bg = "#1e272e"
    text_color = "#f1f1f1"
    metric_bg = "#34495e"
    header_bg = "#2c3e50"
    button_bg = "#2980b9"
else:
    card_bg = "#ffffff"
    text_color = "#333333"
    metric_bg = "#00796b"
    header_bg = "#e8f0f8"
    button_bg = "#1565c0"

# === HEADER ===
st.markdown(
    f"<h2 style='text-align:center; color:{text_color}; background:{header_bg}; padding:15px; border-radius:10px;' >Model Monitoring Dashboard</h2>",
    unsafe_allow_html=True
)

# === LOAD PREDICTIVE MODELS ===
models_df = load_df("""
    SELECT model_id, model_name, version, created_at, performance_metrics
    FROM predictive_models
    ORDER BY created_at DESC
""")

if models_df.empty:
    st.warning("No predictive models found.")
else:
    for _, row in models_df.iterrows():
        metrics_raw = row['performance_metrics']
        metrics_valid = validate_metrics(metrics_raw)
        metrics = json.loads(metrics_raw) if metrics_valid else {}

        precision = f"{metrics.get('precision', 0) * 100:.1f}%" if metrics_valid else "N/A"
        recall = f"{metrics.get('recall', 0) * 100:.1f}%" if metrics_valid else "N/A"
        accuracy = f"{metrics.get('accuracy', 0) * 100:.1f}%" if metrics_valid else "N/A"
        f1 = f"{metrics.get('f1_score', 0) * 100:.1f}%" if metrics_valid else "N/A"

        # Use a container for the card
        with st.container():
            st.markdown(f"<div style='background:{card_bg}; padding:15px; border-radius:12px; color:{text_color}; box-shadow:0 4px 15px rgba(0,0,0,0.2); margin-bottom:15px;'>"
                        f"<h4 style='margin-bottom:5px'>{row['model_name']} (v{row['version']})</h4>"
                        f"<span style='font-size:12px; opacity:0.7'>Created at: {row['created_at']}</span>"
                        f"</div>", unsafe_allow_html=True)
            
            # Metrics in columns for better layout
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Precision", precision)
            col2.metric("Recall", recall)
            col3.metric("Accuracy", accuracy)
            col4.metric("F1 Score", f1)

# === JSON Validator ===
st.markdown("---")
st.subheader("📝 Validate Custom Model Metrics JSON")
input_metrics = st.text_area("Enter JSON:", height=150)

if st.button("Validate"):
    if validate_metrics(input_metrics):
        st.success("✅ Valid performance metrics JSON!")
    else:
        st.error("❌ Invalid or missing required fields: precision, recall, accuracy, f1_score.")
