import os
import sqlite3
from typing import Optional

import pandas as pd
import streamlit as st
import altair as alt

# If you have centralized styling already, use this:
# from utils import inject_global_styles, badge
# For now, keep a small local badge fallback if utils isn't ready.

DB_PATH = "ga_maintenance.db"

st.set_page_config(
    page_title="Component Analytics",
    page_icon="🧩",
    layout="wide",
)

alt.themes.enable("none")


# ----------------------------
# Minimal helpers
# ----------------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def read_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    try:
        with get_conn() as conn:
            return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


def table_exists(name: str) -> bool:
    df = read_sql(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    )
    return not df.empty


def safe_float(x, default=None):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color};">{label}</span>'


# ----------------------------
# Styles (TEMP)
# Replace this with: from utils import inject_global_styles
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
        bg = "#F3F6FB"
        bg2 = "#EEF3FA"
        panel = "rgba(255, 255, 255, 0.96)"
        border = "rgba(15, 23, 42, 0.12)"
        text = "#0F172A"
        muted = "#475569"
        shadow = "0 8px 20px rgba(2, 6, 23, 0.06)"
        input_bg = "rgba(241, 245, 249, 0.98)"
        accent = "#1F6FEB"

    st.markdown(
        f"""
        <style>
          :root {{
            --text: {text};
            --muted: {muted};
            --panel: {panel};
            --border: {border};
            --shadow: {shadow};
            --input: {input_bg};
            --accent: {accent};
          }}

          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, {bg2}, {bg});
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}

          .block-container {{
            max-width: 96vw;
            padding-top: 1.0rem;
            padding-left: 2.0rem;
            padding-right: 2.0rem;
            padding-bottom: 1.6rem;
          }}

          [data-testid="stToolbar"], footer, header {{
            visibility: hidden;
            height: 0;
          }}

          section[data-testid="stSidebar"] {{
            background: var(--input) !important;
            border-right: 1px solid var(--border);
          }}

          section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
            opacity: 1 !important;
          }}

          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            margin-bottom: 14px;
          }}

          .kpiTitle {{
            font-size: 0.88rem;
            color: var(--muted);
            margin-bottom: 6px;
          }}

          .kpiValue {{
            font-size: 1.55rem;
            font-weight: 850;
            color: var(--text);
          }}

          .kpiSub {{
            margin-top: 8px;
            color: var(--muted);
            font-size: 0.88rem;
          }}

          .badge {{
            display:inline-block;
            padding:4px 10px;
            border-radius:999px;
            font-size:0.80rem;
            font-weight:850;
            color:white;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Data access for this page
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def get_components() -> pd.DataFrame:
    if not table_exists("components"):
        return pd.DataFrame()
    return read_sql(
        """
        SELECT component_id, tail_number, name, type, condition,
               remaining_useful_life, last_health_score
        FROM components
        ORDER BY tail_number, type, name
        """
    )


@st.cache_data(show_spinner=False, ttl=60)
def get_component_rul_series(component_id: int) -> pd.DataFrame:
    if not table_exists("component_predictions"):
        return pd.DataFrame()

    # If you have the RPN view, use it. Else fall back.
    if table_exists("component_predictions_with_rpn"):
        return read_sql(
            """
            SELECT prediction_time, predicted_value, confidence,
                   rpn, rpn_calc,
                   fmea_severity, fmea_occurrence_base, fmea_detection_base
            FROM component_predictions_with_rpn
            WHERE component_id = ? AND prediction_type = 'remaining_life'
            ORDER BY prediction_time ASC
            """,
            (component_id,),
        )

    return read_sql(
        """
        SELECT prediction_time, predicted_value, confidence
        FROM component_predictions
        WHERE component_id = ? AND prediction_type = 'remaining_life'
        ORDER BY prediction_time ASC
        """,
        (component_id,),
    )


@st.cache_data(show_spinner=False, ttl=60)
def get_open_alerts(component_id: int) -> pd.DataFrame:
    if not table_exists("alerts"):
        return pd.DataFrame()
    return read_sql(
        """
        SELECT alert_id, severity, message, generated_time
        FROM alerts
        WHERE component_id = ? AND resolved = 0
        ORDER BY
          CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
          generated_time DESC
        LIMIT 15
        """,
        (component_id,),
    )


def compute_risk(open_alerts: pd.DataFrame, health: Optional[float], rul: Optional[float], rpn: Optional[float]):
    # Alerts dominate
    if open_alerts is not None and not open_alerts.empty:
        sev = open_alerts["severity"].astype(str).str.lower().tolist()
        if "critical" in sev:
            return "High", "#DC2626"
        if "warning" in sev:
            return "Medium", "#2563EB"
        return "Low", "#16A34A"

    # RPN thresholds
    if rpn is not None:
        if rpn >= 200:
            return "High", "#DC2626"
        if rpn >= 80:
            return "Medium", "#2563EB"
        return "Low", "#16A34A"

    # Fallback health/RUL
    h = safe_float(health, None)
    r = safe_float(rul, None)
    if h is not None and h < 60:
        return "High", "#DC2626"
    if r is not None and r < 25:
        return "High", "#DC2626"
    if h is not None and h < 75:
        return "Medium", "#2563EB"
    if r is not None and r < 75:
        return "Medium", "#2563EB"
    return "Low", "#16A34A"


# ----------------------------
# UI
# ----------------------------
st.sidebar.markdown("## 🛩️ GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
inject_global_styles(dark_mode)

st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:14px; margin-bottom:10px;">
      <div>
        <div style="font-size:2.0rem; font-weight:900; letter-spacing:-0.02em;">Component Analytics</div>
        <div class="kpiSub">Trends and supporting details for maintenance review.</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

comps = get_components()
if comps.empty:
    st.markdown(
        '<div class="card"><div class="kpiTitle">No data</div><div class="kpiSub">components table not found or empty.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

comps = comps.copy()
comps["label"] = comps.apply(lambda r: f"{r['tail_number']} • {r['type']} • {r['name']} #{r['component_id']}", axis=1)
comp_ids = comps["component_id"].tolist()

st.sidebar.markdown("### Select component")
selected_comp_id = st.sidebar.selectbox(
    "Component",
    options=comp_ids,
    format_func=lambda cid: comps.loc[comps["component_id"] == cid, "label"].iloc[0],
)

row = comps[comps["component_id"] == int(selected_comp_id)].iloc[0]
health = safe_float(row.get("last_health_score"), None)
rul_now = safe_float(row.get("remaining_useful_life"), None)

series = get_component_rul_series(int(selected_comp_id))
alerts_df = get_open_alerts(int(selected_comp_id))

# Latest RPN (if available)
rpn_val = None
sev = occ = det = None
conf_latest = None
last_time = None

if not series.empty:
    tail_row = series.iloc[-1]
    conf_latest = safe_float(tail_row.get("confidence"), None)
    last_time = str(tail_row.get("prediction_time", "")).strip()

    # rpn_calc preferred
    if "rpn_calc" in series.columns and pd.notna(tail_row.get("rpn_calc", None)):
        rpn_val = safe_float(tail_row.get("rpn_calc"), None)
    elif "rpn" in series.columns and pd.notna(tail_row.get("rpn", None)):
        rpn_val = safe_float(tail_row.get("rpn"), None)

    sev = tail_row.get("fmea_severity", None)
    occ = tail_row.get("fmea_occurrence_base", None)
    det = tail_row.get("fmea_detection_base", None)

risk_label, risk_color = compute_risk(alerts_df, health, rul_now, rpn_val)

# Top strip (decision-ready)
c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.1, 1.1, 1.2], gap="large")

with c1:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">Risk</div>
          <div class="kpiValue">{badge(risk_label, risk_color)}</div>
          <div class="kpiSub">Current action level</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">Health</div>
          <div class="kpiValue">{("—" if health is None else f"{int(round(health))}")}</div>
          <div class="kpiSub">0–100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">RUL</div>
          <div class="kpiValue">{("—" if rul_now is None else f"{int(round(rul_now))} h")}</div>
          <div class="kpiSub">Estimated remaining time</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">Confidence</div>
          <div class="kpiValue">{("—" if conf_latest is None else f"{conf_latest:.0%}")}</div>
          <div class="kpiSub">Latest prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">RPN</div>
          <div class="kpiValue">{("—" if rpn_val is None else f"{int(round(rpn_val))}")}</div>
          <div class="kpiSub">FMEA risk priority</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Main area: chart + alerts
left, right = st.columns([2.2, 1.0], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>RUL trend</div>", unsafe_allow_html=True)

    if series.empty:
        st.markdown("<div class='kpiSub'>No RUL history found for this component.</div>", unsafe_allow_html=True)
    else:
        plot_df = series.copy()
        plot_df["prediction_time"] = pd.to_datetime(plot_df["prediction_time"], errors="coerce")
        plot_df = plot_df.dropna(subset=["prediction_time"])

        axis = "#CBD5E1" if dark_mode else "#334155"
        grid_op = 0.15 if dark_mode else 0.25

        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("prediction_time:T", title="Time", axis=alt.Axis(labelColor=axis, titleColor=axis)),
                y=alt.Y("predicted_value:Q", title="RUL (hours)", axis=alt.Axis(labelColor=axis, titleColor=axis)),
                tooltip=[
                    alt.Tooltip("prediction_time:T", title="Time"),
                    alt.Tooltip("predicted_value:Q", title="RUL (h)", format=".0f"),
                    alt.Tooltip("confidence:Q", title="Confidence", format=".0%"),
                ],
            )
            .properties(height=340)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridOpacity=grid_op)
        )

        st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>Component</div>", unsafe_allow_html=True)
    st.markdown(f"**{row['tail_number']} • {row['type']} • {row['name']}**")
    st.markdown(f"<div class='kpiSub'>Condition: <b>{row.get('condition','—')}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpiSub'>Last update: {last_time if last_time else '—'}</div>", unsafe_allow_html=True)

    if rpn_val is not None and sev is not None and occ is not None and det is not None and pd.notna(sev) and pd.notna(occ) and pd.notna(det):
        st.markdown(
            f"<div class='kpiSub'>RPN breakdown: <b>S {int(sev)} × O {int(occ)} × D {int(det)}</b></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>Open alerts</div>", unsafe_allow_html=True)

    if alerts_df.empty:
        st.markdown(f"<div class='kpiSub'>{badge('None', '#16A34A')} No open alerts.</div>", unsafe_allow_html=True)
    else:
        for _, a in alerts_df.iterrows():
            sev = str(a.get("severity", "")).lower()
            if sev == "critical":
                b = badge("Critical", "#DC2626")
            elif sev == "warning":
                b = badge("Warning", "#2563EB")
            else:
                b = badge("Advisory", "#16A34A")

            msg = str(a.get("message", "")).strip()
            ts = str(a.get("generated_time", "")).strip()

            st.markdown(
                f"""
                <div style="margin-top:10px;">
                  {b}
                  <div class="kpiSub" style="margin-top:6px;">{msg}</div>
                  <div class="kpiSub" style="margin-top:4px; opacity:0.8;">{ts}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


# Optional: show raw table only as a collapsible
with st.expander("Show raw records (for audit)"):
    if series.empty:
        st.write("No prediction records.")
    else:
        st.dataframe(series.tail(250), use_container_width=True, hide_index=True)
