import streamlit as st
import pandas as pd
import sqlite3
import altair as alt

# Ensure Altair respects our styling
alt.themes.enable("none")

DB_PATH = "ga_maintenance.db"

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Due Preventive Maintenance",
    page_icon="🛠️",
    layout="wide",
)

# ----------------------------
# GLOBAL STYLES (reuse same theme)
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

          .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            color: white;
            background: {accent};
          }}

          [data-baseweb="select"] > div {{
            border-radius: 12px !important;
            background: {input_bg} !important;
            border: 1px solid {border} !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## 🛠 GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
st.sidebar.caption("Maintenance planning view")

inject_global_styles(dark_mode)

# ----------------------------
# DATA LOADING
# ----------------------------
@st.cache_data(show_spinner=False)
def load_due_tasks():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM due_preventive_tasks
            ORDER BY timestamp DESC
            """,
            conn,
        )

tasks_df = load_due_tasks()

# ----------------------------
# HERO
# ----------------------------
st.markdown(
    """
    <div style="margin-bottom:18px;">
      <h1>Due Preventive Maintenance</h1>
      <div class="muted">
        FAA-aligned preventive maintenance actions requiring review or scheduling.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# EMPTY STATE
# ----------------------------
if tasks_df.empty:
    st.markdown(
        """
        <div class="card">
          <span class="badge">All Clear</span>
          <p style="margin-top:10px;">
            No preventive maintenance tasks are currently due.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------
# FILTER BAR (top, not sidebar)
# ----------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1, 1, 2])

    systems = ["All"] + sorted(tasks_df["system"].dropna().unique().tolist())
    tails = ["All"] + sorted(tasks_df["tail_number"].dropna().unique().tolist())

    with f1:
        selected_system = st.selectbox("System", systems)

    with f2:
        selected_tail = st.selectbox("Tail Number", tails)

    with f3:
        st.markdown(
            f"<span class='badge'>{len(tasks_df)} total tasks</span>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# Apply filters
filtered_df = tasks_df.copy()
if selected_system != "All":
    filtered_df = filtered_df[filtered_df["system"] == selected_system]
if selected_tail != "All":
    filtered_df = filtered_df[filtered_df["tail_number"] == selected_tail]

# ----------------------------
# TASK TABLE
# ----------------------------
st.markdown(
    f"""
    <div class="card">
      <h3>Tasks requiring attention</h3>
      <div class="muted" style="margin-bottom:8px;">
        Filtered view of FAA-aligned preventive maintenance actions.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
)

# ----------------------------
# DOWNLOAD
# ----------------------------
st.download_button(
    label="📥 Download CSV",
    data=filtered_df.to_csv(index=False).encode(),
    file_name="due_preventive_tasks.csv",
    mime="text/csv",
)
