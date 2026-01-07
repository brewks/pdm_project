import os
import sqlite3
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt

import math

from utils import inject_global_styles, badge

# ----------------------------
# CONFIG
# ----------------------------
DB_PATH = "ga_maintenance.db"
SQL_SEED_FILE = "full_pdm_seed.sql"

st.set_page_config(
    page_title="General Aviation Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------
# DB BOOTSTRAP (if missing)
# ----------------------------
def ensure_db_exists():
    if os.path.exists(DB_PATH):
        return
    if not os.path.exists(SQL_SEED_FILE):
        st.error("Database missing and SQL seed file not found.")
        st.stop()

    try:
        with sqlite3.connect(DB_PATH) as conn:
            with open(SQL_SEED_FILE, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
    except Exception as e:
        st.error(f"Database restore failed: {e}")
        st.stop()


@st.cache_resource(show_spinner=False)
def get_conn():
    ensure_db_exists()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(show_spinner=False)
def read_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, get_conn(), params=params)
    except Exception as e:
        st.error("Database query failed. Check that your DB schema matches the expected tables/views.")
        st.caption(f"Details: {type(e).__name__}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def table_exists(name: str) -> bool:
    df = read_sql(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    )
    return not df.empty


def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def to_float_or_none(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def fmt_int_or_dash(x):
    v = to_float_or_none(x)
    return "—" if v is None else str(int(round(v)))


# ----------------------------
# STYLES
# ----------------------------


def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color};">{label}</span>'

# ----------------------------
# DATA LOADERS
# ----------------------------
@st.cache_data(show_spinner=False)
def get_components(tail_number: Optional[str]) -> pd.DataFrame:
    if tail_number and tail_number != "All":
        return read_sql(
            """
            SELECT component_id, tail_number, name, type, condition, remaining_useful_life, last_health_score
            FROM components
            WHERE tail_number = ?
            ORDER BY type, name
            """,
            (tail_number,),
        )
    return read_sql(
        """
        SELECT component_id, tail_number, name, type, condition, remaining_useful_life, last_health_score
        FROM components
        ORDER BY tail_number, type, name
        """
    )


@st.cache_data(show_spinner=False)
def get_dashboard_snapshot() -> pd.DataFrame:
    if not table_exists("dashboard_snapshot_view"):
        return pd.DataFrame()
    return read_sql("SELECT * FROM dashboard_snapshot_view ORDER BY tail_number")


@st.cache_data(show_spinner=False)
def get_open_alerts_for_component(component_id: int) -> pd.DataFrame:
    return read_sql(
        """
        SELECT alert_id, severity, message, generated_time
        FROM alerts
        WHERE component_id = ? AND resolved = 0
        ORDER BY
          CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
          generated_time DESC
        LIMIT 12
        """,
        (component_id,),
    )


@st.cache_data(show_spinner=False)
def get_rul_series(component_id: int) -> pd.DataFrame:
    return read_sql(
        """
        SELECT prediction_time, predicted_value, confidence
        FROM component_predictions
        WHERE component_id = ? AND prediction_type = 'remaining_life'
        ORDER BY prediction_time ASC
        """,
        (component_id,),
    )


@st.cache_data(show_spinner=False)
def get_latest_prediction_with_rpn(component_id: int) -> pd.DataFrame:
    if table_exists("component_predictions_with_rpn"):
        return read_sql(
            """
            SELECT *
            FROM component_predictions_with_rpn
            WHERE component_id = ? AND prediction_type = 'remaining_life'
            ORDER BY prediction_time DESC
            LIMIT 1
            """,
            (component_id,),
        )

    return read_sql(
        """
        SELECT prediction_id, component_id, model_id, prediction_time, prediction_type, predicted_value, confidence, time_horizon
        FROM component_predictions
        WHERE component_id = ? AND prediction_type = 'remaining_life'
        ORDER BY prediction_time DESC
        LIMIT 1
        """,
        (component_id,),
    )


@st.cache_data(show_spinner=False)
def get_model_label(model_id: int) -> str:
    df = read_sql(
        """
        SELECT model_name, version, model_type
        FROM predictive_models
        WHERE model_id = ?
        """,
        (model_id,),
    )
    if df.empty:
        return "Model: —"
    r = df.iloc[0]
    v = str(r.get("version", "")).strip()
    mt = str(r.get("model_type", "")).strip()
    name = str(r.get("model_name", "")).strip()
    suffix = f" ({mt})" if mt else ""
    ver = f" v{v}" if v else ""
    return f"{name}{ver}{suffix}" if name else "Model: —"


def compute_risk_label(open_alerts: pd.DataFrame, health: float, rul_hours: float, rpn_val: Optional[float]) -> Tuple[str, str]:
    if not open_alerts.empty:
        sev = open_alerts["severity"].astype(str).str.lower().tolist()
        if "critical" in sev:
            return "High", "#DC2626"
        if "warning" in sev:
            return "Medium", "#2563EB"
        return "Low", "#16A34A"

    if rpn_val is not None:
        if rpn_val >= 200:
            return "High", "#DC2626"
        if rpn_val >= 80:
            return "Medium", "#2563EB"
        return "Low", "#16A34A"

    if health < 60 or (rul_hours is not None and rul_hours < 25):
        return "High", "#DC2626"
    if health < 75 or (rul_hours is not None and rul_hours < 75):
        return "Medium", "#2563EB"
    return "Low", "#16A34A"


def top_kpi_strip(snapshot: pd.DataFrame):
    if snapshot.empty:
        st.markdown(
            '<div class="card kpi"><div class="kpiTitle">Fleet</div><div class="kpiValue">No snapshot data</div>'
            '<div class="kpiSub">dashboard_snapshot_view not available</div></div>',
            unsafe_allow_html=True,
        )
        return

    df = snapshot.copy()

    active_alerts = int(df["active_alerts"].fillna(0).sum()) if "active_alerts" in df.columns else 0

    status_rank = {"maintenance_required": 4, "attention_needed": 3, "monitoring": 2, "normal": 1}
    worst = "normal"
    if "predictive_status" in df.columns and not df["predictive_status"].isna().all():
        worst = max(df["predictive_status"].astype(str).str.lower().tolist(), key=lambda x: status_rank.get(x, 0))

    avg_health = None
    if "health_score" in df.columns:
        hs = pd.to_numeric(df["health_score"], errors="coerce")
        if hs.notna().any():
            avg_health = float(hs.mean())

    avg_health_text = fmt_int_or_dash(avg_health)
    avg_health_val = 0.0 if to_float_or_none(avg_health) is None else float(avg_health)
    avg_health_val = max(0.0, min(100.0, avg_health_val))

    last_ts = ""
    if "last_prediction_time" in df.columns:
        s = df["last_prediction_time"].dropna()
        last_ts = str(s.max()) if not s.empty else ""

    if worst in ("maintenance_required",):
        status_badge = badge("Maintenance required", "#DC2626")
    elif worst in ("attention_needed",):
        status_badge = badge("Attention needed", "#F59E0B")
    elif worst in ("monitoring",):
        status_badge = badge("Monitoring", "#2563EB")
    else:
        status_badge = badge("Normal", "#16A34A")

    c1, c2, c3, c4 = st.columns([1.15, 1.30, 1.05, 1.05])

    with c1:
        st.markdown(
            f"""
            <div class="card kpi">
              <div class="kpiTitle">Active alerts</div>
              <div class="kpiValue">{active_alerts}</div>
              <div class="kpiSub">Unresolved across fleet</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card kpi">
              <div class="kpiTitle">Average health</div>
              <div class="gaugeWrap">
                <div class="gauge" style="--deg:{int((avg_health_val/100)*360)}deg;">
                  <div class="gaugeInner">
                    <div class="gaugeVal">{avg_health_text}</div>
                    <div class="gaugeLbl">0–100</div>
                  </div>
                </div>
                <div>
                  <div class="kpiSub">Higher is better</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="card kpi">
              <div class="kpiTitle">Last analysis</div>
              <div class="kpiValue">{last_ts if last_ts else "—"}</div>
              <div class="kpiSub">Latest recorded run</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="card kpi">
              <div class="kpiTitle">Overall status</div>
              <div class="kpiValue">{status_badge}</div>
              <div class="kpiSub">Decision-ready summary</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def rul_trend_chart(series: pd.DataFrame, dark_mode: bool):
    if series.empty:
        st.markdown(
            '<div class="card"><div class="kpiTitle">RUL trend</div><div class="kpiSub">No RUL history for this component.</div></div>',
            unsafe_allow_html=True,
        )
        return

    df = series.copy()
    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    df = df.dropna(subset=["prediction_time"])

    axis_color = "#A8B3C7" if dark_mode else "#334155"
    grid_op = 0.15 if dark_mode else 0.25

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("prediction_time:T", title="Time", axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
            y=alt.Y("predicted_value:Q", title="Remaining time (hours)", axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
            tooltip=[
                alt.Tooltip("prediction_time:T", title="Time"),
                alt.Tooltip("predicted_value:Q", title="RUL (hrs)", format=".0f"),
                alt.Tooltip("confidence:Q", title="Confidence", format=".0%"),
            ],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure_axis(gridOpacity=grid_op)
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>RUL trend</div>", unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_open_alerts(alerts_df: pd.DataFrame):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>Open alerts</div>", unsafe_allow_html=True)

    if alerts_df.empty:
        st.markdown(f"<div class='kpiSub'>{badge('None', '#16A34A')} No open alerts for this component.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for _, r in alerts_df.iterrows():
        sev = str(r.get("severity", "")).lower()
        if sev == "critical":
            b = badge("Critical", "#DC2626")
        elif sev == "warning":
            b = badge("Warning", "#2563EB")
        else:
            b = badge("Advisory", "#16A34A")

        msg = str(r.get("message", "")).strip()
        ts = str(r.get("generated_time", "")).strip()
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


def main():
    # Sidebar controls (NO aircraft selectbox here)
    st.sidebar.markdown("### GA PdM")
    dark_mode = st.sidebar.toggle("Dark mode", value=True)
    inject_global_styles(dark_mode)

    try:
        st.sidebar.page_link("pages/01_Maintenance_Tasks.py", label="Maintenance Tasks", icon="🛠️")
    except Exception:
        st.sidebar.caption("🛠️ Maintenance Tasks (use sidebar page list)")

    mode = st.sidebar.radio("View", ["Pilot / Operator", "Maintenance / Engineer"], index=0)

    # Header
    st.markdown(
        """
        <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:14px;">
          <div>
            <div style="font-size:2.05rem; font-weight:900; letter-spacing:-0.02em;">General Aviation Predictive Maintenance</div>
            <div class="muted">Clear health, risk, and maintenance signals for pilots and mechanics.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI strip (fleet-level)
    snapshot = get_dashboard_snapshot()
    top_kpi_strip(snapshot)

    # ----------------------------
    # MAIN PAGE: AIRCRAFT + COMPONENT SELECTOR (2-step)
    # ----------------------------
    all_components = get_components(None)
    if all_components.empty:
        st.markdown(
            '<div class="card"><div class="kpiTitle">No components</div>'
            '<div class="kpiSub">No component records found in the system.</div></div>',
            unsafe_allow_html=True,
        )
        return

    tail_numbers = sorted(all_components["tail_number"].dropna().unique().tolist())
    selected_tail = st.selectbox("Aircraft (tail number)", ["All"] + tail_numbers, index=0)

    comps = get_components(None if selected_tail == "All" else selected_tail)
    if comps.empty:
        st.markdown(
            '<div class="card"><div class="kpiTitle">No components</div>'
            '<div class="kpiSub">No component records found for the selected aircraft.</div></div>',
            unsafe_allow_html=True,
        )
        return

    types = sorted(comps["type"].dropna().unique().tolist())
    selected_type = st.selectbox("Component category", ["All"] + types, index=0)
    if selected_type != "All":
        comps = comps[comps["type"] == selected_type].copy()

    comps = comps.reset_index(drop=True)

    def component_label(i: int) -> str:
        r = comps.loc[i]
        typ = str(r["type"]).replace("_", " ").title()
        name = str(r["name"])
        if selected_tail == "All":
            return f"{r['tail_number']} • {typ} • {name}  (#{int(r['component_id'])})"
        return f"{typ} • {name}  (#{int(r['component_id'])})"

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='kpiTitle'>Component</div>", unsafe_allow_html=True)

        selected_idx = st.selectbox(
            " ",
            options=list(range(len(comps))),
            format_func=component_label,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        selected_comp_id = int(comps.loc[selected_idx, "component_id"])

        series = get_rul_series(selected_comp_id)
        rul_trend_chart(series, dark_mode)

    with right:
        row = comps.loc[selected_idx]
        health = safe_float(row.get("last_health_score"), 0.0)
        rul_hours = safe_float(row.get("remaining_useful_life"), 0.0)

        alerts_df = get_open_alerts_for_component(selected_comp_id)
        latest = get_latest_prediction_with_rpn(selected_comp_id)

        model_line = "Model: —"
        conf = None
        last_pred_time = ""
        rpn = None
        rpn_calc = None
        sev = occ = det = None

        if not latest.empty:
            latest_row = latest.iloc[0]
            conf = safe_float(latest_row.get("confidence"), None)
            last_pred_time = str(latest_row.get("prediction_time", "")).strip()
            model_id = latest_row.get("model_id", None)
            if model_id is not None:
                model_line = get_model_label(int(model_id))

            if "rpn_calc" in latest.columns:
                rpn_calc = latest_row.get("rpn_calc", None)
            if "rpn" in latest.columns:
                rpn = latest_row.get("rpn", None)

            sev = latest_row.get("fmea_severity", None)
            occ = latest_row.get("fmea_occurrence_base", None)
            det = latest_row.get("fmea_detection_base", None)

        rpn_to_show = None
        if rpn_calc is not None and pd.notna(rpn_calc):
            rpn_to_show = safe_float(rpn_calc, None)
        elif rpn is not None and pd.notna(rpn):
            rpn_to_show = safe_float(rpn, None)

        risk_label, risk_color = compute_risk_label(alerts_df, health, rul_hours, rpn_to_show)

        st.markdown(
            f"""
            <div class="card">
              <div class="kpiTitle">Summary</div>
              <div style="font-weight:850; margin-top:4px;">{row['tail_number']} • {row['type']} • {row['name']}</div>
              <div class="kpiSub" style="margin-top:10px;">Condition: <b>{row.get('condition','—')}</b></div>
              <div class="kpiSub" style="margin-top:8px;">Risk: {badge(risk_label, risk_color)}</div>
              <div class="kpiSub" style="margin-top:10px;">{model_line}</div>
              <div class="kpiSub" style="margin-top:6px;">Last update: {last_pred_time if last_pred_time else "—"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_open_alerts(alerts_df)


if __name__ == "__main__":
    main()

