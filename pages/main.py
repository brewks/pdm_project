import streamlit as st
import json
import altair as alt
from utils import load_df, validate_metrics

# Ensure Altair respects custom styling
alt.themes.enable("none")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📊",
    layout="wide",
)

# ----------------------------
# GLOBAL STYLES (same Airbus theme)
# ----------------------------
def inject_global_styles(dark_mode: bool):
    if dark_mode:
        bg = "#0B1220"
        bg2 = "#0E172A"
        panel = "rgba(17, 27, 47, 0.86)"
        border = "rgba(148, 163, 184, 0.14)"
        text = "rgba(226, 232, 240, 0.92)"
        muted = "rgba(226, 232, 240, 0.68)"
        shadow = "0 10px 28px rgba(0,0,0,0.35)"
        input_bg = "rgba(15, 23, 42, 0.75)"
        accent = "#5AA2FF"
    else:
        bg = "#F3F6FB"
        bg2 = "#EEF3FA"
        panel = "rgba(255, 255, 255, 0.88)"
        border = "rgba(15, 23, 42, 0.10)"
        text = "rgba(15, 23, 42, 0.92)"
        muted = "rgba(15, 23, 42, 0.65)"
        shadow = "0 10px 26px rgba(2, 6, 23, 0.08)"
        input_bg = "rgba(255, 255, 255, 0.92)"
        accent = "#1F6FEB"

    st.markdown(
        f"""
        <style>
          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, {bg2}, {bg});
            color: {text};
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}

          .block-container {{
            max-width: 1200px;
            padding-top: 1.25rem;
          }}

          [data-testid="stToolbar"], footer, header {{
            visibility: hidden;
            height: 0;
          }}

          h1, h2, h3 {{
            letter-spacing: -0.02em;
          }}

          .muted {{
            color: {muted};
            font-size: 0.95rem;
          }}

          .card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: {shadow};
            margin-bottom: 16px;
          }}

          .kpiTitle {{
            font-size: 0.88rem;
            color: {muted};
          }}
          .kpiValue {{
            font-size: 1.45rem;
            font-weight: 700;
          }}

          .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
            background: {accent};
          }}

          textarea.stTextArea textarea {{
            background: {input_bg};
            color: {text};
            border: 1px solid {border};
            border-radius: 12px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">{title}</div>
          <div class="kpiValue">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## 📊 GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
st.sidebar.caption("Model performance & governance")

inject_global_styles(dark_mode)

# ----------------------------
# HERO
# ----------------------------
st.markdown(
    """
    <div style="margin-bottom:18px;">
      <h1>Model Monitoring</h1>
      <div class="muted">
        Performance, versioning, and validation status of deployed predictive models.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# LOAD MODELS
# ----------------------------
models_df = load_df(
    """
    SELECT model_id, model_name, version, created_at, performance_metrics
    FROM predictive_models
    ORDER BY created_at DESC
    """
)

if models_df.empty:
    st.markdown(
        """
        <div class="card">
          <span class="badge">No Models</span>
          <p style="margin-top:10px;">
            No predictive models are currently registered in the system.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------
# MODEL CARDS
# ----------------------------
for _, row in models_df.iterrows():
    metrics_raw = row["performance_metrics"]
    metrics_valid = validate_metrics(metrics_raw)
    metrics = json.loads(metrics_raw) if metrics_valid else {}

    precision = f"{metrics.get('precision', 0) * 100:.1f}%" if metrics_valid else "N/A"
    recall = f"{metrics.get('recall', 0) * 100:.1f}%" if metrics_valid else "N/A"
    accuracy = f"{metrics.get('accuracy', 0) * 100:.1f}%" if metrics_valid else "N/A"
    f1 = f"{metrics.get('f1_score', 0) * 100:.1f}%" if metrics_valid else "N/A"

    st.markdown(
        f"""
        <div class="card">
          <h3 style="margin-bottom:4px;">
            {row['model_name']} <span class="muted">v{row['version']}</span>
          </h3>
          <div class="muted">
            Created: {row['created_at']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1: kpi_card("Precision", precision)
    with c2: kpi_card("Recall", recall)
    with c3: kpi_card("Accuracy", accuracy)
    with c4: kpi_card("F1 Score", f1)

# ----------------------------
# JSON VALIDATOR
# ----------------------------
st.markdown(
    """
    <div class="card">
      <h3>Validate Model Metrics JSON</h3>
      <div class="muted">
        Paste custom metrics JSON to verify required fields.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_metrics = st.text_area(
    "Performance Metrics JSON",
    height=160,
    label_visibility="collapsed",
)

if st.button("Validate"):
    if validate_metrics(input_metrics):
        st.success("✅ Valid performance metrics JSON.")
    else:
        st.error("❌ Invalid or missing required fields: precision, recall, accuracy, f1_score.")
