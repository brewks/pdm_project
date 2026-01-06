import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import time

# ----------------------------
# CONFIG
# ----------------------------
DB_PATH = "ga_maintenance.db"

st.set_page_config(
    page_title="GA Predictive Maintenance",
    page_icon="🛩️",
    layout="wide",
)

# ----------------------------
# GLOBAL STYLES (Airbus-like, FIXED sidebar contrast)
# ----------------------------
def inject_global_styles(dark_mode: bool):
    if dark_mode:
        bg = "#0B1220"
        bg2 = "#0E172A"
        panel = "rgba(17, 27, 47, 0.86)"
        border = "rgba(148, 163, 184, 0.14)"
        text = "#E5E7EB"
        muted = "#CBD5E1"
        shadow = "0 10px 26px rgba(0,0,0,0.30)"
        input_bg = "rgba(15, 23, 42, 0.85)"
        accent = "#5AA2FF"
    else:
        # ✅ STRONG readable light-mode colors
        bg = "#F3F6FB"
        bg2 = "#EEF3FA"
        panel = "rgba(255, 255, 255, 0.96)"
        border = "rgba(15, 23, 42, 0.12)"
        text = "#0F172A"       # solid slate
        muted = "#475569"
        shadow = "0 8px 20px rgba(2, 6, 23, 0.06)"
        input_bg = "rgba(241, 245, 249, 0.98)"  # subtle slate sidebar
        accent = "#1F6FEB"

    st.markdown(
        f"""
        <style>
          :root {{
            --text: {text};
          }}

          /* App background */
          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, {bg2}, {bg});
            color: {text};
            font-family: ui-sans-serif, system-ui, -apple-system,
                         Segoe UI, Roboto, Helvetica, Arial;
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
            color: {text};
          }}

          .muted {{
            color: {muted};
            font-size: 0.95rem;
          }}

          /* Cards */
          .card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: {shadow};
            margin-bottom: 16px;
            color: {text};
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

          /* Sidebar background */
          section[data-testid="stSidebar"] {{
            background: {input_bg} !important;
            border-right: 1px solid {border};
          }}

          /* ============================
             SIDEBAR CONTRAST FIX
             ============================ */
          section[data-testid="stSidebar"] * {{
            color: {text} !important;
            opacity: 1 !important;
          }}

          section[data-testid="stSidebar"] a,
          section[data-testid="stSidebar"] a span,
          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {{
            color: {text} !important;
            opacity: 1 !important;
          }}

          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] p,
          section[data-testid="stSidebar"] .stMarkdown,
          section[data-testid="stSidebar"] .stMarkdown * {{
            color: {text} !important;
            opacity: 1 !important;
          }}

          section[data-testid="stSidebar"] [data-baseweb] * {{
            color: {text} !important;
            opacity: 1 !important;
          }}

          section[data-testid="stSidebar"] [data-testid="stSlider"] * {{
            color: {text} !important;
          }}

          /* DataFrame text */
          .stDataFrame, .stDataFrame div {{
            color: {text} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## 🛩️ GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)

inject_global_styles(dark_mode)

# ----------------------------
# Matplotlib contrast sync
# ----------------------------
if dark_mode:
    plt.rcParams.update({
        "text.color": "#E5E7EB",
        "axes.labelcolor": "#E5E7EB",
        "xtick.color": "#CBD5E1",
        "ytick.color": "#CBD5E1",
        "axes.edgecolor": "#64748B",
    })
else:
    plt.rcParams.update({
        "text.color": "#0F172A",
        "axes.labelcolor": "#0F172A",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "axes.edgecolor": "#334155",
    })

# ----------------------------
# DATA LOADER
# ----------------------------
def load_data(query):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)

# ----------------------------
# HERO
# ----------------------------
st.markdown(
    """
    <div style="margin-bottom:18px;">
      <h1>General Aviation Predictive Maintenance</h1>
      <div class="muted">
        Fleet health monitoring, predictive insights, and maintenance prioritization.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 10)

view_choice = st.sidebar.radio(
    "View",
    [
        "Components Needing Attention",
        "Dashboard Snapshot",
        "Latest Predictions",
        "Engine Health Overview",
    ],
)

# ----------------------------
# LOAD DATA
# ----------------------------
if view_choice == "Components Needing Attention":
    df = load_data("SELECT * FROM components_needing_attention;")
elif view_choice == "Dashboard Snapshot":
    df = load_data("SELECT * FROM dashboard_snapshot_view;")
elif view_choice == "Engine Health Overview":
    df = load_data("SELECT * FROM engine_health_view;")
else:
    df = load_data(
        "SELECT * FROM component_predictions ORDER BY prediction_time DESC LIMIT 100;"
    )

# ----------------------------
# SECTION HEADER
# ----------------------------
st.markdown(
    f"""
    <div class="card">
      <h3>{view_choice}</h3>
      <div class="muted">{len(df)} records loaded</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# DATA TABLE
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# VISUALS
# ----------------------------
def plot_rul_bar(df):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df["component_id"].astype(str), df["predicted_value"], color="#5AA2FF")
    ax.set_title("Latest Component RUL")
    ax.set_ylabel("RUL (hours)")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_rul_trend(df):
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 4))
    for cid, g in df.groupby("component_id"):
        ax.plot(g["prediction_time"], g["predicted_value"], marker="o", alpha=0.7)
    ax.set_title("RUL Trend Over Time")
    ax.set_ylabel("RUL (hours)")
    ax.grid(alpha=0.2)
    st.pyplot(fig)

if not df.empty and view_choice == "Latest Predictions":
    plot_rul_bar(df)
    plot_rul_trend(df)

# ----------------------------
# ALERTS
# ----------------------------
if "confidence" in df.columns and "prediction_type" in df.columns:
    critical = df[(df["confidence"] > 0.9) & (df["prediction_type"] == "failure")]
    if not critical.empty:
        st.error(f"🚨 {len(critical)} critical failure predictions detected")

# ----------------------------
# CONFIDENCE FILTER + EXPORT
# ----------------------------
if "confidence" in df.columns:
    conf_level = st.slider("Minimum confidence", 0.0, 1.0, 0.7)
    filtered = df[df["confidence"] >= conf_level]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### Filtered Predictions (≥ {conf_level})")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    if not filtered.empty:
        st.download_button(
            "Download CSV",
            filtered.to_csv(index=False).encode(),
            file_name="filtered_predictions.csv",
            mime="text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# AUTO REFRESH
# ----------------------------
if refresh_interval > 0:
    st.info(f"Auto-refreshing every {refresh_interval}s")
    time.sleep(refresh_interval)
    st.rerun()
