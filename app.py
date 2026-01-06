import os
import json
import sqlite3
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt

# Optional (recommended): gauges + radial charts
# pip install plotly
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Ensure Altair doesn't override styling
alt.themes.enable("none")

# ----------------------------
# CONFIGURATION
# ----------------------------
DB_PATH = "ga_maintenance.db"
SQL_SEED_FILE = "full_pdm_seed.sql"
LOGO_PATH = "logo.png"


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="GA Predictive Maintenance",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# STYLES (Production-ish)
# ----------------------------
def inject_global_styles(dark_mode: bool):
    """
    Key goals:
    - High contrast in BOTH modes (fixes invisible sidebar text in light mode)
    - Calm, aviation-grade palette
    - Clean cards, consistent inputs, readable tables
    """
    if dark_mode:
        bg = "#0B1220"
        bg2 = "#0E172A"
        sidebar_bg = "#081022"
        panel = "rgba(17, 27, 47, 0.86)"
        border = "rgba(148, 163, 184, 0.16)"
        text = "rgba(226, 232, 240, 0.94)"
        muted = "rgba(226, 232, 240, 0.70)"
        shadow = "0 10px 28px rgba(0,0,0,0.35)"
        input_bg = "rgba(15, 23, 42, 0.78)"
        accent = "#5AA2FF"
        grid = "rgba(148, 163, 184, 0.12)"
        sidebar_text = "rgba(226, 232, 240, 0.94)"
        sidebar_muted = "rgba(226, 232, 240, 0.74)"
        table_bg = "rgba(15, 23, 42, 0.30)"
    else:
        bg = "#F3F6FB"
        bg2 = "#EEF3FA"
        sidebar_bg = "#FFFFFF"
        panel = "rgba(255, 255, 255, 0.92)"
        border = "rgba(15, 23, 42, 0.10)"
        text = "rgba(15, 23, 42, 0.94)"
        muted = "rgba(15, 23, 42, 0.70)"
        shadow = "0 12px 26px rgba(2, 6, 23, 0.10)"
        input_bg = "rgba(255, 255, 255, 0.96)"
        accent = "#1F6FEB"
        grid = "rgba(15, 23, 42, 0.10)"
        # IMPORTANT: force readable sidebar text in light mode
        sidebar_text = "rgba(15, 23, 42, 0.92)"
        sidebar_muted = "rgba(15, 23, 42, 0.70)"
        table_bg = "rgba(15, 23, 42, 0.03)"

    st.markdown(
        f"""
        <style>
          :root {{
            --bg: {bg};
            --bg2: {bg2};
            --sidebar_bg: {sidebar_bg};
            --panel: {panel};
            --border: {border};
            --text: {text};
            --muted: {muted};
            --shadow: {shadow};
            --input: {input_bg};
            --accent: {accent};
            --grid: {grid};
            --sidebar_text: {sidebar_text};
            --sidebar_muted: {sidebar_muted};
            --table_bg: {table_bg};
          }}

          /* App background */
          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, var(--bg2), var(--bg));
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}

          .block-container {{
            padding-top: 1.10rem;
            padding-bottom: 2.0rem;
            max-width: 1250px;
          }}

          /* Hide Streamlit chrome */
          [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          /* Sidebar: force contrast in light mode */
          section[data-testid="stSidebar"] {{
            background: var(--sidebar_bg) !important;
            border-right: 1px solid var(--border);
          }}
          section[data-testid="stSidebar"] * {{
            color: var(--sidebar_text) !important;
          }}
          section[data-testid="stSidebar"] .stCaption,
          section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: var(--sidebar_muted) !important;
          }}
          /* Sidebar inputs */
          section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: var(--input) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
          }}
          section[data-testid="stSidebar"] .stRadio > div {{
            background: transparent !important;
          }}

          /* Typography */
          h1, h2, h3 {{ letter-spacing: -0.02em; }}
          .muted {{
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.35rem;
          }}

          /* Cards */
          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(8px);
          }}
          .cardTitle {{
            font-size: 0.95rem;
            color: var(--muted);
            margin-bottom: 8px;
          }}
          .kpiValue {{
            font-size: 1.55rem;
            font-weight: 750;
            color: var(--text);
          }}
          .kpiSub {{
            margin-top: 6px;
            color: var(--muted);
            font-size: 0.88rem;
          }}

          /* Badges */
          .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            color: white;
            vertical-align: middle;
          }}

          /* Inputs */
          [data-baseweb="select"] > div {{
            border-radius: 12px !important;
            background: var(--input) !important;
            border: 1px solid var(--border) !important;
          }}
          textarea.stTextArea textarea {{
            background: var(--input);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 12px;
            font-size: 14px;
          }}

          /* Buttons */
          .stButton > button {{
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            font-weight: 700;
            border: 1px solid var(--border);
            background: var(--accent);
            color: white;
          }}
          .stButton > button:hover {{ filter: brightness(0.97); }}

          /* DataFrame background blend */
          [data-testid="stDataFrame"] {{
            background: var(--table_bg) !important;
            border-radius: 14px;
            border: 1px solid var(--border);
          }}

          /* Tabs */
          .stTabs [data-baseweb="tab"] {{
            color: var(--muted) !important;
          }}
          .stTabs [aria-selected="true"] {{
            color: var(--text) !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, level: str) -> str:
    colors = {
        "ok":  "#1F9D55",
        "warn":"#DFAF2C",
        "crit":"#D64545",
        "info":"#3B82F6",
        "muted":"#64748B",
    }
    c = colors.get(level, "#64748B")
    return f'<span class="badge" style="background:{c};">{text}</span>'


def card_kpi(title: str, value: str, sub: str | None = None):
    sub_html = f'<div class="kpiSub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="cardTitle">{title}</div>
          <div class="kpiValue">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_dt(x) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(x, errors="coerce")
        return None if pd.isna(ts) else ts
    except Exception:
        return None


# ----------------------------
# AUTOMATIC DB RESTORATION IF MISSING
# ----------------------------
if not os.path.exists(DB_PATH):
    st.warning("Database file not found. Attempting to restore from SQL seed...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            with open(SQL_SEED_FILE, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
        st.success("Database successfully restored.")
    except Exception as e:
        st.error(f"Database restoration failed: {e}")


# ----------------------------
# DB HELPERS
# ----------------------------
@st.cache_data(show_spinner=False, ttl=45)
def load_df(query: str, params: tuple | None = None) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=45)
def load_scalar(query: str, params: tuple | None = None):
    df = load_df(query, params=params)
    if df.empty:
        return None
    return df.iloc[0, 0]


# ----------------------------
# SIDEBAR (Operator controls)
# ----------------------------
st.sidebar.markdown("## 🛩️ GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
inject_global_styles(dark_mode)

st.sidebar.markdown("---")
refresh = st.sidebar.slider("Auto-refresh (seconds)", 0, 60, 10)
if refresh > 0:
    # simple refresh without extra dependencies
    st.sidebar.caption(f"Auto-refresh is ON ({refresh}s)")
    st.sidebar.markdown(
        f"<meta http-equiv='refresh' content='{int(refresh)}'>",
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
show_evidence = st.sidebar.toggle("USCIS demo mode (clean labels)", value=True)
st.sidebar.caption("Keeps wording simple for pilots, mechanics, and reviewers.")

# ----------------------------
# LOAD DATA (Fleet-level)
# ----------------------------
snapshot_df = load_df("SELECT * FROM dashboard_snapshot_view;")
attention_df = load_df("SELECT * FROM components_needing_attention;")
due_tasks_df = load_df("SELECT * FROM due_preventive_tasks ORDER BY timestamp DESC;")
open_alerts_df = load_df("SELECT * FROM alerts WHERE resolved = 0 ORDER BY generated_time DESC;")

# For selectors
aircraft_df = load_df("SELECT tail_number, model, predictive_status FROM aircraft ORDER BY tail_number;")
components_df = load_df(
    """
    SELECT component_id, tail_number, name, type, condition, remaining_useful_life, last_health_score
    FROM components
    ORDER BY tail_number, type, name;
    """
)

if aircraft_df.empty or components_df.empty:
    st.error("No aircraft/components available. Database may not have loaded correctly.")
    st.stop()

tails = aircraft_df["tail_number"].dropna().unique().tolist()
selected_tail = st.sidebar.selectbox("Tail number", tails)

comp_rows = components_df[components_df["tail_number"] == selected_tail].copy()
comp_rows["label"] = comp_rows.apply(lambda r: f"{r['type']} — {r['name']}", axis=1)
comp_label_to_id = dict(zip(comp_rows["label"], comp_rows["component_id"]))

selected_comp_label = st.sidebar.selectbox("Component", comp_rows["label"].tolist())
selected_comp_id = int(comp_label_to_id[selected_comp_label])

# Pull component-specific predictions (latest first)
preds_df = load_df(
    """
    SELECT *
    FROM component_predictions
    WHERE component_id = ?
    ORDER BY prediction_time DESC
    """,
    params=(selected_comp_id,),
)

# ----------------------------
# HEADER (Title + logo at top-right)
# ----------------------------
h_left, h_right = st.columns([10, 2], vertical_alignment="center")
with h_left:
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
          <h1 style="margin:0; padding:0;">General Aviation Predictive Maintenance</h1>
          <div class="muted" style="margin-top:6px;">
            Practical health + risk signals for flight ops and maintenance planning.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h_right:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        # silent in demo mode; minimal warning otherwise
        if not show_evidence:
            st.caption("logo.png not found in app directory.")


# ----------------------------
# FLEET AT A GLANCE (Non-technical KPIs)
# ----------------------------
fleet_aircraft = int(snapshot_df["tail_number"].nunique()) if not snapshot_df.empty else int(aircraft_df["tail_number"].nunique())
fleet_attention = int(attention_df["component_id"].nunique()) if not attention_df.empty else 0
fleet_open_alerts = int(open_alerts_df.shape[0]) if not open_alerts_df.empty else 0
fleet_due_tasks = int(due_tasks_df.shape[0]) if not due_tasks_df.empty else 0

k1, k2, k3, k4 = st.columns(4, gap="large")
with k1:
    card_kpi("Aircraft monitored", f"{fleet_aircraft}", "Fleet-level overview")
with k2:
    card_kpi("Items needing attention", f"{fleet_attention}", "Low RUL, low health, or high-risk prediction")
with k3:
    card_kpi("Open alerts", f"{fleet_open_alerts}", "Unresolved advisories / warnings / critical")
with k4:
    card_kpi("Due preventive tasks", f"{fleet_due_tasks}", "From FAA-aligned recommendations")

st.markdown("")

# ----------------------------
# SELECTED AIRCRAFT SUMMARY
# ----------------------------
snap_row = snapshot_df[snapshot_df["tail_number"] == selected_tail].copy()
model = None
predictive_status = None
last_pred_time = None
last_rul = None
last_conf = None
health_score = None
active_alerts = None

if not snap_row.empty:
    r = snap_row.iloc[0]
    model = r.get("model", None)
    predictive_status = r.get("predictive_status", None)
    last_pred_time = r.get("last_prediction_time", None)
    last_rul = r.get("last_rul_prediction", None)
    last_conf = r.get("prediction_confidence", None)
    health_score = r.get("health_score", None)
    active_alerts = r.get("active_alerts", None)

# derive simple labels
def status_badge(status_text: str | None, alerts_count: int | None):
    if alerts_count and int(alerts_count) > 0:
        return badge("Open alerts", "warn")
    if status_text and str(status_text).strip().lower() in {"critical", "warning"}:
        return badge(str(status_text).title(), "warn")
    return badge("Normal", "ok")

sum_left, sum_right = st.columns([2.2, 1], gap="large")

with sum_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    title = f"{selected_tail}" + (f" — {model}" if model else "")
    st.markdown(f"### {title}")

    dt = safe_dt(last_pred_time)
    dt_str = dt.strftime("%Y-%m-%d %H:%M") if dt else "N/A"

    rul_str = f"{float(last_rul):.1f} h" if last_rul is not None and pd.notna(last_rul) else "N/A"
    conf_str = f"{float(last_conf)*100:.0f}%" if last_conf is not None and pd.notna(last_conf) else "N/A"
    hs_str = f"{int(health_score)}" if health_score is not None and pd.notna(health_score) else "N/A"

    st.markdown(
        f"""
        <div class="muted" style="margin-top:-6px;">
          Last update: <b>{dt_str}</b>
        </div>
        <div style="margin-top:10px;">
          {status_badge(predictive_status, active_alerts)}&nbsp;&nbsp;
          <span class="muted">Confidence (latest): <b>{conf_str}</b></span>
        </div>
        <div style="margin-top:12px;" class="muted">
          <b>Health (0–100):</b> {hs_str}&nbsp;&nbsp; • &nbsp;&nbsp;
          <b>Estimated time to service:</b> {rul_str}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top attention list for this tail
    tail_attention = attention_df[attention_df["tail_number"] == selected_tail].copy() if not attention_df.empty else pd.DataFrame()
    if not tail_attention.empty:
        tail_attention = tail_attention.head(6)
        st.markdown("#### What needs attention (this aircraft)")
        show_cols = [c for c in ["name", "type", "last_health_score", "prediction_type", "predicted_value", "confidence", "time_horizon"] if c in tail_attention.columns]
        st.dataframe(tail_attention[show_cols], use_container_width=True, hide_index=True)
    else:
        st.markdown("#### What needs attention (this aircraft)")
        st.markdown(badge("Normal", "ok") + " No flagged components right now.", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with sum_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Selected component")
    st.markdown(f"**{selected_comp_label}**")

    comp_row = comp_rows[comp_rows["component_id"] == selected_comp_id]
    comp_cond = "unknown"
    comp_rul = None
    comp_hs = None
    if not comp_row.empty:
        rr = comp_row.iloc[0]
        comp_cond = str(rr.get("condition", "unknown"))
        comp_rul = rr.get("remaining_useful_life", None)
        comp_hs = rr.get("last_health_score", None)

    comp_rul_str = f"{float(comp_rul):.1f} h" if comp_rul is not None and pd.notna(comp_rul) else "N/A"
    comp_hs_str = f"{int(comp_hs)}" if comp_hs is not None and pd.notna(comp_hs) else "N/A"

    st.markdown(f"Condition: **{comp_cond}**")
    st.markdown(f"Health: **{comp_hs_str} / 100**")
    st.markdown(f"Estimated time to service: **{comp_rul_str}**")

    # Component alert (from alerts table and/or predictions)
    comp_open_alerts = open_alerts_df[open_alerts_df["component_id"] == selected_comp_id].copy() if not open_alerts_df.empty else pd.DataFrame()
    crit_pred = pd.DataFrame()
    if not preds_df.empty and "prediction_type" in preds_df.columns and "confidence" in preds_df.columns:
        crit_pred = preds_df[(preds_df["prediction_type"] == "failure") & (preds_df["confidence"] >= 0.90)].copy()

    st.markdown("---")
    st.markdown("### Alerts")
    if not comp_open_alerts.empty:
        top = comp_open_alerts.iloc[0]
        sev = str(top.get("severity", "advisory")).lower()
        level = "crit" if sev == "critical" else ("warn" if sev in {"warning"} else "info")
        msg = str(top.get("message", ""))
        st.markdown(badge(sev.title(), level) + f" {msg}", unsafe_allow_html=True)
    elif not crit_pred.empty:
        row = crit_pred.iloc[0]
        hz = row.get("time_horizon", "N/A")
        cf = float(row.get("confidence", 0.0)) * 100
        st.markdown(badge("Critical", "crit") + " Predicted failure risk", unsafe_allow_html=True)
        st.markdown(f"- Horizon: **{hz}**")
        st.markdown(f"- Confidence: **{cf:.0f}%**")
    else:
        st.markdown(badge("Normal", "ok") + " No open alerts for this component.", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# VISUALS + ACTIONS (Operator-friendly)
# ----------------------------
v_left, v_right = st.columns([2.2, 1], gap="large")

with v_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Trend: estimated time to service (RUL)")

    rul_series = pd.DataFrame()
    if not preds_df.empty:
        rul_series = preds_df[preds_df["prediction_type"] == "remaining_life"].copy()

    if not rul_series.empty:
        rul_series["prediction_time"] = pd.to_datetime(rul_series["prediction_time"], errors="coerce")
        rul_series = rul_series.dropna(subset=["prediction_time"]).sort_values("prediction_time")

        axis_label = "#A8B3C7" if dark_mode else "#334155"

        chart = (
            alt.Chart(rul_series)
            .mark_line(point=True)
            .encode(
                x=alt.X("prediction_time:T", title="Time"),
                y=alt.Y("predicted_value:Q", title="Hours"),
                tooltip=[
                    alt.Tooltip("prediction_time:T", title="Time"),
                    alt.Tooltip("predicted_value:Q", title="Estimated hours"),
                    alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                ],
            )
            .properties(height=340)
            .configure_view(strokeOpacity=0)
            .configure_axis(
                grid=True,
                gridOpacity=0.18,
                labelColor=axis_label,
                titleColor=axis_label,
                tickColor="rgba(0,0,0,0)",
            )
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No RUL trend available yet for this component.")

    # quick, relatable "what-if" slider (does not claim certainty)
    st.markdown("#### Quick scenario check")
    colA, colB, colC = st.columns(3)
    with colA:
        usage = st.slider("Utilization", 0.5, 1.5, 1.0, 0.05)  # lighter/heavier use
    with colB:
        severity = st.slider("Operating severity", 0.7, 1.6, 1.0, 0.05)  # rough ops/hot env/etc
    with colC:
        buffer = st.slider("Safety buffer", 0.0, 50.0, 10.0, 1.0)

    base_rul = float(comp_rul) if comp_rul is not None and pd.notna(comp_rul) else None
    if base_rul is not None:
        projected = max((base_rul / max(usage * severity, 0.01)) - buffer, 0.0)
        st.markdown(
            f"""
            <div class="muted">
              Baseline estimate: <b>{base_rul:.1f} h</b> • Scenario estimate: <b>{projected:.1f} h</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Scenario check needs a baseline RUL value for this component.")

    st.markdown("</div>", unsafe_allow_html=True)

with v_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Gauges")

    # derive gauge values
    hs_val = float(comp_hs) if comp_hs is not None and pd.notna(comp_hs) else 0.0
    conf_val = float(preds_df["confidence"].iloc[0]) * 100 if not preds_df.empty and "confidence" in preds_df.columns and pd.notna(preds_df["confidence"].iloc[0]) else 0.0

    if PLOTLY_OK:
        g1 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=hs_val,
                number={"suffix": " / 100"},
                title={"text": "Health"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1F6FEB"},
                    "steps": [
                        {"range": [0, 40], "color": "rgba(214, 69, 69, 0.25)"},
                        {"range": [40, 70], "color": "rgba(223, 175, 44, 0.20)"},
                        {"range": [70, 100], "color": "rgba(31, 157, 85, 0.18)"},
                    ],
                },
            )
        )
        g1.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g1, use_container_width=True)

        g2 = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=conf_val,
                number={"suffix": "%"},
                title={"text": "Latest confidence"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#5AA2FF"},
                    "steps": [
                        {"range": [0, 60], "color": "rgba(223, 175, 44, 0.20)"},
                        {"range": [60, 100], "color": "rgba(31, 157, 85, 0.18)"},
                    ],
                },
            )
        )
        g2.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(g2, use_container_width=True)
    else:
        st.info("Install Plotly for gauges: `pip install plotly`")

    st.markdown("---")
    st.markdown("### Recommended next steps")

    steps = []
    # Use clear, non-alarming phrasing
    if not comp_open_alerts.empty:
        sev = str(comp_open_alerts.iloc[0].get("severity", "advisory")).lower()
        if sev == "critical":
            steps.append(("crit", "Schedule inspection before next extended operation window."))
        elif sev == "warning":
            steps.append(("warn", "Plan inspection in the next maintenance slot."))
        else:
            steps.append(("info", "Monitor trend and confirm at next inspection."))

    if base_rul is not None:
        if base_rul < 25:
            steps.append(("warn", "Prioritize this component in the next shop visit (low time-to-service)."))
        elif base_rul < 50:
            steps.append(("info", "Place on watch list; re-check after next flights."))
        else:
            steps.append(("ok", "No immediate action needed; continue normal monitoring."))

    if due_tasks_df is not None and not due_tasks_df.empty:
        # show tail-specific tasks
        tail_tasks = due_tasks_df[due_tasks_df["tail_number"] == selected_tail].copy()
        if not tail_tasks.empty:
            steps.append(("warn", f"{len(tail_tasks)} preventive task(s) recorded as due for this aircraft."))

    if not steps:
        steps = [("ok", "No action suggested at this time.")]

    for lvl, txt in steps[:4]:
        st.markdown(f"- {badge(lvl.title() if lvl!='ok' else 'Normal', lvl)} {txt}", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# DETAILS (Operator drill-down)
# ----------------------------
tabs = st.tabs(
    [
        "Alerts",
        "Due preventive tasks",
        "Latest predictions",
        "Engine overview",
        "Explainability (for review)",
    ]
)

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Open alerts")

    if open_alerts_df.empty:
        st.markdown(badge("Normal", "ok") + " No open alerts in the system.", unsafe_allow_html=True)
    else:
        # filter for selected tail + selected component
        f1, f2 = st.columns(2)
        with f1:
            scope_tail = st.checkbox("Show only selected aircraft", value=True)
        with f2:
            scope_comp = st.checkbox("Show only selected component", value=False)

        df = open_alerts_df.copy()
        if scope_tail:
            df = df[df["tail_number"] == selected_tail]
        if scope_comp:
            df = df[df["component_id"] == selected_comp_id]

        show_cols = [c for c in ["generated_time", "severity", "alert_type", "message", "notification_status"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Due preventive tasks")

    if due_tasks_df.empty:
        st.markdown(badge("Normal", "ok") + " No due preventive tasks recorded.", unsafe_allow_html=True)
    else:
        df = due_tasks_df.copy()
        # focus on the selected tail by default
        show_tail_only = st.checkbox("Show only selected aircraft", value=True, key="due_tail_only")
        if show_tail_only:
            df = df[df["tail_number"] == selected_tail]

        # readable columns
        cols = [c for c in ["timestamp", "tail_number", "component_name", "task_name", "system", "ac_43_ref", "confidence"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download tasks (CSV)",
            data=df[cols].to_csv(index=False).encode("utf-8"),
            file_name=f"due_tasks_{selected_tail}.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Latest predictions for selected component")

    if preds_df.empty:
        st.info("No predictions available for this component yet.")
    else:
        df = preds_df.copy()
        # friendly labels
        if show_evidence:
            df["prediction_type"] = df["prediction_type"].replace(
                {
                    "remaining_life": "time_to_service",
                    "performance_degradation": "performance",
                    "maintenance_need": "maintenance_need",
                    "failure": "failure_risk",
                }
            )
        cols = [c for c in ["prediction_time", "prediction_type", "predicted_value", "confidence", "time_horizon"] if c in df.columns]
        st.dataframe(df[cols].head(50), use_container_width=True, hide_index=True)

        st.download_button(
            "Download predictions (CSV)",
            data=df[cols].to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{selected_tail}_component_{selected_comp_id}.csv",
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Engine overview (selected aircraft)")

    eng = load_df(
        "SELECT * FROM engine_health_view WHERE tail_number = ? ORDER BY timestamp DESC LIMIT 200;",
        params=(selected_tail,),
    )
    if eng.empty:
        st.info("No engine overview records found for this aircraft.")
    else:
        # Keep it practical: show the core engine parameters + predicted remaining life if present
        show_cols = [c for c in eng.columns if c in {
            "timestamp",
            "max_cht", "max_egt", "max_oil_temp", "max_oil_press",
            "max_rpm", "max_fuel_flow", "max_manifold_press", "max_coolant_temp",
            "remaining_life", "prediction_confidence"
        }]
        st.dataframe(eng[show_cols], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Explainability (for review)")

    if preds_df.empty:
        st.info("No prediction records available.")
    else:
        row = preds_df.iloc[0]

        # Pull the fields your schema supports: explanation + feature_vector_summary + model_input_features
        explanation = str(row.get("explanation", "") or "").strip()
        fvec = str(row.get("feature_vector_summary", "") or "").strip()
        feats = str(row.get("model_input_features", "") or "").strip()

        if show_evidence:
            st.caption("This section helps a reviewer understand what the system is doing (without code).")

        if explanation:
            st.markdown("**Why the system flagged this:**")
            st.write(explanation)
        else:
            st.markdown(badge("Info", "info") + " No explanation text stored for the latest prediction.", unsafe_allow_html=True)

        if fvec:
            st.markdown("**Input summary used by the model:**")
            st.write(fvec)

        if feats:
            st.markdown("**Input feature list (stored in DB):**")
            st.code(feats, language="text")

    st.markdown("</div>", unsafe_allow_html=True)
