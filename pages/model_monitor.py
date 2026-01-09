# utils.py
import sqlite3
import pandas as pd
import json
import streamlit as st

DB_PATH = "ga_maintenance.db"


# ----------------------------
# DB HELPERS (keep)
# ----------------------------
def load_df(query, params: tuple = ()):
    """Run a SQL query and return a pandas DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


def validate_metrics(metrics_json):
    """Validate that a JSON string includes all required performance metric fields."""
    try:
        data = json.loads(metrics_json)
        required = ["precision", "recall", "accuracy", "f1_score"]
        return all(k in data for k in required)
    except (json.JSONDecodeError, TypeError):
        return False


def table_exists(name: str) -> bool:
    """True if a table/view exists in SQLite."""
    df = load_df(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    )
    return not df.empty


# ----------------------------
# UI HELPERS (centralized)
# ----------------------------
def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color};">{label}</span>'


def kpi_card(title: str, value: str, sub: str = "") -> str:
    """Returns KPI card HTML (use with st.markdown(..., unsafe_allow_html=True))."""
    sub_html = f"<div class='kpiSub'>{sub}</div>" if sub else ""
    return f"""
    <div class="card kpi">
      <div class="kpiTitle">{title}</div>
      <div class="kpiValue">{value}</div>
      {sub_html}
    </div>
    """


def altair_axis_colors(dark_mode: bool):
    """Central axis/grid styling for Altair charts."""
    axis_color = "#A8B3C7" if dark_mode else "#334155"
    grid_opacity = 0.15 if dark_mode else 0.25
    return axis_color, grid_opacity


def inject_global_styles(dark_mode: bool):
    """
    ONE source of truth for your UI theme.
    Call this once per page after you know dark_mode.

    IMPORTANT for multipage apps:
    - Do NOT fully hide Streamlit header/toolbar, because the sidebar toggle lives there.
    - Instead we keep it visible but subtle, so the sidebar never becomes "unreachable".
    """
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
        sidebar_bg = "linear-gradient(180deg, #081024 0%, #0B1220 100%)"
        sidebar_border = "1px solid rgba(148,163,184,0.12)"
    else:
        bg = "#F3F6FB"
        bg2 = "#EEF3FA"
        panel = "rgba(255, 255, 255, 0.92)"
        border = "rgba(15, 23, 42, 0.10)"
        text = "rgba(15, 23, 42, 0.92)"
        muted = "rgba(15, 23, 42, 0.65)"
        shadow = "0 10px 26px rgba(2, 6, 23, 0.08)"
        input_bg = "rgba(255, 255, 255, 0.96)"
        accent = "#1F6FEB"
        sidebar_bg = "linear-gradient(180deg, #F7FAFF 0%, #F3F6FB 100%)"
        sidebar_border = "1px solid rgba(2,6,23,0.10)"

    st.markdown(
        f"""
        <style>
          :root {{
            --bg: {bg};
            --bg2: {bg2};
            --panel: {panel};
            --border: {border};
            --text: {text};
            --muted: {muted};
            --shadow: {shadow};
            --input: {input_bg};
            --accent: {accent};
          }}

          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, var(--bg2), var(--bg));
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system,
                         Segoe UI, Roboto, Helvetica, Arial;
          }}

          /* Layout */
          .block-container {{
            padding-top: 1.0rem;
            padding-bottom: 1.6rem;
            max-width: 96vw;
            padding-left: 2.0rem;
            padding-right: 2.0rem;
          }}

          footer {{ visibility: hidden; }}

          /* Keep header/toolbar */
          [data-testid="stHeader"] {{
            background: transparent !important;
          }}
          header {{
            visibility: visible !important;
          }}

          [data-testid="stToolbar"] {{
            visibility: visible !important;
            height: auto !important;
            opacity: 0.12;
            transition: opacity 0.15s ease;
          }}
          [data-testid="stToolbar"]:hover {{
            opacity: 1;
          }}

          [data-testid="stDecoration"] {{
            display: none !important;
          }}

          /* Sidebar */
          section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: {sidebar_border};
          }}

          section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
            opacity: 1 !important;
          }}
          section[data-testid="stSidebar"] .stCaption,
          section[data-testid="stSidebar"] small,
          section[data-testid="stSidebar"] label {{
            color: var(--muted) !important;
          }}

          /* Cards */
          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(8px);
          }}

          /* KPI cards */
          .card.kpi {{
            padding: 18px 20px;
            min-height: 104px;
          }}

          .kpiTitle {{
            font-size: 0.88rem;
            color: var(--muted);
            margin-bottom: 6px;
          }}

          .kpiValue {{
            font-size: 1.70rem;
            font-weight: 850;
          }}

          .kpiSub {{
            margin-top: 8px;
            color: var(--muted);
            font-size: 0.88rem;
          }}

          .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.80rem;
            font-weight: 850;
            color: white;
          }}

          /* iOS-like select / inputs */
          [data-baseweb="select"] > div {{
            border-radius: 14px !important;
            background: var(--input) !important;
            border: 1px solid rgba(120,120,140,0.28) !important;
            min-height: 44px !important;
            padding-left: 10px !important;
            padding-right: 10px !important;
            box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
          }}

          [data-baseweb="select"] span,
          [data-baseweb="select"] div {{
            font-size: 16px !important;
          }}

          ul[role="listbox"] {{
            border-radius: 14px !important;
            border: 1px solid rgba(120,120,140,0.22) !important;
            background: var(--panel) !important;
            overflow: hidden !important;
            box-shadow: var(--shadow) !important;
          }}

          ul[role="listbox"] li {{
            font-size: 16px !important;
            padding-top: 10px !important;
            padding-bottom: 10px !important;
          }}

          ul[role="listbox"] li:hover {{
            background: rgba(90,162,255,0.12) !important;
          }}

          [data-baseweb="input"] > div {{
            border-radius: 14px !important;
            background: var(--input) !important;
            border: 1px solid rgba(120,120,140,0.28) !important;
            min-height: 44px !important;
          }}

          /* Buttons */
          .stButton > button {{
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            font-weight: 850;
            border: 1px solid var(--border);
            background: var(--accent);
            color: white;
          }}
          .stButton > button:hover {{
            filter: brightness(0.96);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )
