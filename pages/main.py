import streamlit as st
import json
from utils import load_df, validate_metrics

# === DARK & LIGHT MODE COLOR THEMES ===
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")

if dark_mode:
    # Rich dark palette
    card_bg = "#111827"        # Deep navy charcoal
    text_color = "#E5E7EB"     # Soft silver-white
    metric_bg = "#1F2937"      # Slightly lighter section
    header_bg = "#0F172A"      # Ultra-deep blue-black
    button_bg = "#3B82F6"      # Bright blue accent
    accent_color = "#22D3EE"   # Electric cyan
else:
    # Modern bright light palette
    card_bg = "#FFFFFF"        # Clean white
    text_color = "#1F2937"     # Slate text
    metric_bg = "#A7F3D0"      # Mint highlight
    header_bg = "#EFF6FF"      # Soft pastel blue
    button_bg = "#2563EB"      # Strong blue
    accent_color = "#059669"   # Emerald accent

# === BUTTON STYLE ===
st.markdown(
    f"""
    <style>
    .stButton>button {{
        background-color:{button_bg};
        color:white;
        padding:0.6em 1.5em;
        border-radius:8px;
        border:none;
        font-weight:600;
    }}
    .stButton>button:hover {{
        background-color:{accent_color};
        color:black;
        transition:0.2s;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# === HEADER ===
st.markdown(
    f"""
    <h2 style='
        text-align:center;
        color:{text_color};
        background:{header_bg};
        padding:15px;
        border-radius:10px;
        border-left:6px solid {accent_color};
    '>
        Model Monitoring Dashboard
    </h2>
    """,
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

        # === MODEL CARD ===
        st.markdown(
            f"""
            <div style='
                background:{card_bg};
                padding:18px;
                border-radius:12px;
                color:{text_color};
                box-shadow:0 4px 20px rgba(0,0,0,0.25);
                border-left:4px solid {accent_color};
                margin-bottom:15px;
            '>
                <h4 style='margin-bottom:5px'>{row['model_name']} (v{row['version']})</h4>
                <span style='font-size:12px; opacity:0.75'>Created at: {row['created_at']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # === METRICS (4 COLUMNS) ===
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precision", precision)
        col2.metric("Recall", recall)
        col3.metric("Accuracy", accuracy)
        col4.metric("F1 Score", f1)

# === JSON VALIDATOR SECTION ===
st.markdown("---")
st.subheader("📝 Validate Custom Model Metrics JSON")
input_metrics = st.text_area("Enter JSON:", height=150)

if st.button("Validate JSON"):
    if validate_metrics(input_metrics):
        st.success("✅ Valid performance metrics JSON!")
    else:
        st.error("❌ Invalid or missing required fields: precision, recall, accuracy, f1_score.")
