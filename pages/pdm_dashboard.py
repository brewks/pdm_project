import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import time

# === CONFIG ===
DB_PATH = "ga_maintenance.db"

# Dark plotting theme
plt.style.use("dark_background")
sns.set_theme(style="whitegrid")

# === FUNCTIONS ===
def load_data(query):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def plot_rul_bar(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x=df["component_id"].astype(str),
        y="predicted_value",
        ax=ax,
        palette="Blues",
    )
    ax.set_xlabel("Component ID")
    ax.set_ylabel("Predicted RUL (hours)")
    ax.set_title("Latest Component RUL Predictions")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_confidence_rul(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x="component_id",
        y="predicted_value",
        size="confidence",
        hue="confidence",
        palette="coolwarm",
        ax=ax,
        sizes=(50, 300),
    )
    ax.set_title("RUL vs Component ID (Size = Confidence)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_rul_trend(df):
    if "prediction_time" in df.columns:
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(
            data=df,
            x="prediction_time",
            y="predicted_value",
            hue="component_id",
            marker="o",
            ax=ax,
        )
        ax.set_title("RUL Predictions Over Time")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# === APP LAYOUT ===
st.set_page_config(page_title="GA PdM Dashboard", layout="wide")

# === PREMIUM MODERN DARK THEME ===
st.markdown(
    """
<style>

:root {
    --bg-main: #020617;
    --text-main: #e5e7eb;
    --text-soft: #93c5fd;
    --card-border: rgba(51, 65, 85, 0.6);
    --card-bg: rgba(2, 6, 23, 0.65);
    --hover-glow: rgba(59, 130, 246, 0.25);
}

/* App background */
.stApp {
    background: radial-gradient(circle at top, #0a0f24 0%, #020617 70%);
    color: var(--text-main);
    font-family: 'Segoe UI', sans-serif;
}

/* Main content container */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0b1228 100%);
    color: var(--text-main);
    border-right: 1px solid rgba(30, 64, 175, 0.7);
    box-shadow: 4px 0 25px rgba(0, 0, 0, 0.75);
}

/* Sidebar elements */
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stRadio {
    padding: 12px 14px 10px 14px;
    margin-bottom: 20px;
    border-radius: 14px;
    background: rgba(2, 6, 23, 0.55);
    border: 1px solid var(--card-border);
    backdrop-filter: blur(4px);
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: var(--text-main);
    font-weight: 600;
}

/* Header */
.dashboard-title {
    font-size: 32px;
    font-weight: 700;
    color: var(--text-main);
    text-shadow: 0px 0px 18px rgba(59, 130, 246, 0.7);
}
.big-font {
    font-size: 17px;
    color: var(--text-soft);
}

/* Cards */
.main-card {
    background: var(--card-bg);
    border-radius: 20px;
    padding: 22px 24px;
    border: 1px solid var(--card-border);
    box-shadow: 0 25px 55px rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(6px);
    transition: 0.25s ease-in-out;
}

.main-card:hover {
    border: 1px solid rgba(59, 130, 246, 0.5);
    box-shadow: 0 35px 65px rgba(0, 0, 0, 0.85),
                0 0 20px var(--hover-glow);
}

/* Dataframe background fix */
.stDataFrame {
    background: transparent !important;
}

</style>
""",
    unsafe_allow_html=True,
)

# === HEADER ===
col1, col2 = st.columns([1, 10])
with col1:
    st.image("logo.png", width=80)
with col2:
    st.markdown(
        '<div class="dashboard-title">General Aviation Predictive Maintenance Dashboard</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<p class="big-font">Live aircraft system health, predictive maintenance insights, and automated alerts</p>',
    unsafe_allow_html=True,
)

# === SIDEBAR CONTROLS ===
refresh_interval = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 10)

view_choice = st.sidebar.radio(
    "Select View",
    [
        "Components Needing Attention",
        "Dashboard Snapshot",
        "Latest Predictions",
        "Engine Health Overview",
    ],
)

# === LOAD DATA ===
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

# === DISPLAY DATA ===
st.markdown(
    f"""
<div class="main-card">
    <h4 style="font-size:21px; font-weight:600; margin-bottom:6px;">
        {view_choice}
    </h4>
    <p style="font-size:16px; font-weight:400; margin-top:0; color:#cbd5f5;">
        Data Summary: {len(df)} records loaded
    </p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-card" style="margin-top:12px;">', unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# === CHARTS & LOGIC ===
if df.empty:
    st.warning("⚠ No data available for this view.")
else:

    if view_choice == "Latest Predictions":
        st.write("### Predicted Remaining Useful Life (RUL)")
        plot_rul_bar(df)

        st.write("### RUL vs Component ID with Confidence Levels")
        plot_confidence_rul(df)

        st.write("### Time-Series RUL Trends")
        plot_rul_trend(df)

    if "confidence" in df.columns and "prediction_type" in df.columns:
        critical_alerts = df[
            (df["confidence"] > 0.9) & (df["prediction_type"] == "failure")
        ]
        if not critical_alerts.empty:
            st.error(f"🚨 {len(critical_alerts)} CRITICAL failure predictions detected!")

    if "confidence" in df.columns:
        conf_level = st.slider("Minimum Confidence Threshold", 0.0, 1.0, 0.7)
        filtered_df = df[df["confidence"] >= conf_level]
        st.write(f"### Filtered Predictions (Confidence ≥ {conf_level})")
        st.dataframe(filtered_df, use_container_width=True)

        if not filtered_df.empty:
            st.download_button(
                "⬇️ Download Filtered Data",
                filtered_df.to_csv(index=False).encode(),
                file_name="filtered_predictions.csv",
                mime="text/csv",
            )

# === AUTO-REFRESH ===
if refresh_interval > 0:
    st.info(f"⏳ Auto-refreshing every {refresh_interval} seconds...")
    time.sleep(refresh_interval)
    st.experimental_rerun()
