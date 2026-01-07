import os
import sqlite3
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt


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
        # Streamlit cloud redacts details in UI; show a clean “what to check”.
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


# ----------------------------
# STYLES (YOUR FUNCTION + 2 TINY FIXES)
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
        grid = "rgba(148, 163, 184, 0.10)"
        sidebar_bg = "linear-gradient(180deg, #081024 0%, #0B1220 100%)"
        sidebar_border = "1px solid rgba(148,163,184,0.12)"
        axis_label = "#A8B3C7"
        grid_opacity = 0.15
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
        grid = "rgba(15, 23, 42, 0.08)"
        sidebar_bg = "linear-gradient(180deg, #F7FAFF 0%, #F3F6FB 100%)"
        sidebar_border = "1px solid rgba(2,6,23,0.10)"
        axis_label = "#334155"
        grid_opacity = 0.25

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
            --grid: {grid};
          }}

          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, var(--bg2), var(--bg));
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}

          /* Wider canvas: fixes unused side space */
          .block-container {{
            padding-top: 1.0rem;
            padding-bottom: 1.6rem;
            max-width: 96vw;
            padding-left: 2.0rem;
            padding-right: 2.0rem;
          }}

          [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: {sidebar_border};
          }}

          /* Sidebar spacing + cleaner look */
         section[data-testid="stSidebar"] .block-container {
         padding-top: 1.0rem;
         }
         
          /* -----------------------------
             SIDEBAR TEXT FIX (light mode)
             ----------------------------- */

          /* Force all sidebar text to be readable */
          section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
          }}

          /* Muted sidebar captions / helper text */
          section[data-testid="stSidebar"] .stCaption,
          section[data-testid="stSidebar"] small,
          section[data-testid="stSidebar"] label {{
            color: var(--muted) !important;
          }}

          /* Radio / checkbox / toggle labels */
          section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
          section[data-testid="stSidebar"] [role="radiogroup"] *,
          section[data-testid="stSidebar"] [data-testid="stCheckbox"] *,
          section[data-testid="stSidebar"] [data-testid="stToggle"] * {{
            color: var(--text) !important;
          }}

          /* Selectbox input text */
          section[data-testid="stSidebar"] [data-baseweb="select"] * {{
            color: var(--text) !important;
          }}

          /* Fix the selectbox placeholder in light mode */
          section[data-testid="stSidebar"] [data-baseweb="select"] input {{
            -webkit-text-fill-color: var(--text) !important;
            caret-color: var(--text) !important;
          }}

          /* Slider labels + values */
          section[data-testid="stSidebar"] [data-testid="stSlider"] * {{
            color: var(--text) !important;
          }}

          
          h1, h2, h3 {{ letter-spacing: -0.02em; }}
          .muted {{
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.35rem;
          }}

          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(8px);
          }}

          /* KPI cards: bigger + more "Airbus strip" */
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
            color: var(--text);
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

          /* Inputs */
          [data-baseweb="select"] > div {{
            border-radius: 12px !important;
            background: var(--input) !important;
            border: 1px solid var(--border) !important;
          }}
          [data-baseweb="input"] > div {{
            border-radius: 12px !important;
            background: var(--input) !important;
            border: 1px solid var(--border) !important;
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

          /* Gauges */
          .gaugeWrap {{
            display:flex;
            gap:12px;
            align-items:center;
          }}
          .gauge {{
            width: 84px;
            height: 84px;
            border-radius: 50%;
            background:
              conic-gradient(var(--accent) var(--deg), rgba(148,163,184,0.18) 0);
            display:flex;
            align-items:center;
            justify-content:center;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
          }}
          .gaugeInner {{
            width: 66px;
            height: 66px;
            border-radius: 50%;
            background: var(--panel);
            display:flex;
            align-items:center;
            justify-content:center;
            flex-direction:column;
          }}
          .gaugeVal {{
            font-weight: 900;
            font-size: 1.05rem;
            color: var(--text);
            line-height: 1.1rem;
          }}
          .gaugeLbl {{
            font-size: 0.72rem;
            color: var(--muted);
            margin-top: 2px;
          }}

          /* Tighter spacing between blocks (fixes "too much space") */
          [data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]) {{
            margin-bottom: 0.45rem;
          }}
          [data-testid="stHorizontalBlock"] {{
            gap: 0.95rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(label: str, color: str) -> str:
    return f'<span class="badge" style="background:{color};">{label}</span>'


def gauge_html(title: str, value: float, label: str, pct_0_to_100: bool = True) -> str:
    """
    Renders a clean radial gauge. No stray closing tags -> no </div> artifacts.
    """
    if pct_0_to_100:
        v = max(0.0, min(100.0, value))
        deg = int((v / 100.0) * 360)
        shown = f"{int(round(v))}"
        sub = label
    else:
        # for non-percent, show numeric and still map to 0..100 for ring if needed externally
        v = value
        deg = 0
        shown = f"{v:.0f}"
        sub = label

    return f"""
    <div class="card">
      <div class="kpiTitle">{title}</div>
      <div class="gaugeWrap">
        <div class="gauge" style="--deg:{deg}deg;">
          <div class="gaugeInner">
            <div class="gaugeVal">{shown}</div>
            <div class="gaugeLbl">{sub}</div>
          </div>
        </div>
      </div>
    </div>
    """


# ----------------------------
# DATA LOADERS
# ----------------------------
@st.cache_data(show_spinner=False)
def get_aircraft_list() -> pd.DataFrame:
    return read_sql("SELECT tail_number, model, predictive_status FROM aircraft ORDER BY tail_number")


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
    # Uses the view you created in schm_tbls.txt
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

    # fallback: no view
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


# ----------------------------
# RISK LOGIC (PILOT-FRIENDLY)
# ----------------------------
def compute_risk_label(open_alerts: pd.DataFrame, health: float, rul_hours: float, rpn_val: Optional[float]) -> Tuple[str, str]:
    """
    Returns (risk_label, color).
    Priority: alerts -> RPN -> health/rul.
    """
    # Alerts dominate
    if not open_alerts.empty:
        sev = open_alerts["severity"].astype(str).str.lower().tolist()
        if "critical" in sev:
            return "High", "#DC2626"
        if "warning" in sev:
            return "Medium", "#2563EB"
        return "Low", "#16A34A"

    # RPN thresholds (typical FMEA ranges; keep simple)
    if rpn_val is not None:
        if rpn_val >= 200:
            return "High", "#DC2626"
        if rpn_val >= 80:
            return "Medium", "#2563EB"
        return "Low", "#16A34A"

    # Health / RUL fallback
    if health < 60 or (rul_hours is not None and rul_hours < 25):
        return "High", "#DC2626"
    if health < 75 or (rul_hours is not None and rul_hours < 75):
        return "Medium", "#2563EB"
    return "Low", "#16A34A"


# ----------------------------
# UI SECTIONS
# ----------------------------
def top_kpi_strip(snapshot: pd.DataFrame, tail_filter: str):
    """
    Top strip for quick “fleet awareness”.
    """
    if snapshot.empty:
        st.markdown('<div class="card kpi"><div class="kpiTitle">Fleet</div><div class="kpiValue">No snapshot data</div><div class="kpiSub">dashboard_snapshot_view not available</div></div>', unsafe_allow_html=True)
        return

    df = snapshot.copy()
    if tail_filter != "All":
        df = df[df["tail_number"] == tail_filter]

    active_alerts = int(df["active_alerts"].fillna(0).sum()) if "active_alerts" in df.columns else 0
    # Worst status across filtered aircraft
    status_rank = {"maintenance_required": 4, "attention_needed": 3, "monitoring": 2, "normal": 1}
    worst = "normal"
    if "predictive_status" in df.columns and not df["predictive_status"].isna().all():
        worst = max(df["predictive_status"].astype(str).str.lower().tolist(), key=lambda x: status_rank.get(x, 0))

    # Average health score
    avg_health = safe_float(df["health_score"].dropna().mean(), 0.0) if "health_score" in df.columns else 0.0

    # Last analysis time
    last_ts = ""
    if "last_prediction_time" in df.columns:
        s = df["last_prediction_time"].dropna()
        last_ts = str(s.max()) if not s.empty else ""

    # Status badge
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
              <div class="kpiSub">Unresolved across selected aircraft</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        # avg health gauge
        st.markdown(
            f"""
            <div class="card kpi">
              <div class="kpiTitle">Average health</div>
              <div class="gaugeWrap">
                <div class="gauge" style="--deg:{int((max(0,min(100,avg_health))/100)*360)}deg;">
                  <div class="gaugeInner">
                    <div class="gaugeVal">{int(round(avg_health))}</div>
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
        st.markdown('<div class="card"><div class="kpiTitle">RUL trend</div><div class="kpiSub">No RUL history for this component.</div></div>', unsafe_allow_html=True)
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

    # Small, readable list
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
    # Sidebar controls
    st.sidebar.markdown("### GA PdM")
    dark_mode = st.sidebar.toggle("Dark mode", value=True)
    inject_global_styles(dark_mode)

    mode = st.sidebar.radio(
        "View",
        ["Pilot / Operator", "Maintenance / Engineer"],
        index=0,
    )

    aircraft_df = get_aircraft_list()
    tail_options = ["All"] + (aircraft_df["tail_number"].tolist() if not aircraft_df.empty else [])
    tail = st.sidebar.selectbox("Aircraft (tail number)", tail_options, index=0)

    # Header (clean, no fluff)
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
    top_kpi_strip(snapshot, tail)

    # Component selector (NO label->id dict => NO KeyError)
    comps = get_components(tail)
    if comps.empty:
        st.markdown('<div class="card"><div class="kpiTitle">No components</div><div class="kpiSub">No component records found for the selected aircraft.</div></div>', unsafe_allow_html=True)
        return

    # Build labels for display only
    comps = comps.copy()
    comps["label"] = comps.apply(lambda r: f"{r['tail_number']} • {r['type']} • {r['name']} #{r['component_id']}", axis=1)

    comp_ids = comps["component_id"].tolist()

    left, right = st.columns([1.55, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='kpiTitle'>Component</div>", unsafe_allow_html=True)

        selected_comp_id = st.selectbox(
            " ",
            options=comp_ids,
            format_func=lambda cid: comps.loc[comps["component_id"] == cid, "label"].iloc[0],
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # RUL trend
        series = get_rul_series(int(selected_comp_id))
        rul_trend_chart(series, dark_mode)

    with right:
        # Summary panel data
        row = comps[comps["component_id"] == int(selected_comp_id)].iloc[0]
        health = safe_float(row.get("last_health_score"), 0.0)
        rul_hours = safe_float(row.get("remaining_useful_life"), 0.0)

        alerts_df = get_open_alerts_for_component(int(selected_comp_id))
        latest = get_latest_prediction_with_rpn(int(selected_comp_id))

        # Latest prediction details
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

            # RPN
            if "rpn_calc" in latest.columns:
                rpn_calc = latest_row.get("rpn_calc", None)
            if "rpn" in latest.columns:
                rpn = latest_row.get("rpn", None)
            # S/O/D
            sev = latest_row.get("fmea_severity", None)
            occ = latest_row.get("fmea_occurrence_base", None)
            det = latest_row.get("fmea_detection_base", None)

        # Choose RPN value to show
        rpn_to_show = None
        if rpn_calc is not None and pd.notna(rpn_calc):
            rpn_to_show = safe_float(rpn_calc, None)
        elif rpn is not None and pd.notna(rpn):
            rpn_to_show = safe_float(rpn, None)

        # Risk label
        risk_label, risk_color = compute_risk_label(alerts_df, health, rul_hours, rpn_to_show)

        # Summary card (no fluff)
        st.markdown(
            f"""
            <div class="card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px;">
                <div>
                  <div class="kpiTitle">Summary</div>
                  <div style="font-weight:850; margin-top:4px;">{row['tail_number']} • {row['type']} • {row['name']}</div>
                  <div class="kpiSub" style="margin-top:10px;">Condition: <b>{row.get('condition','—')}</b></div>
                  <div class="kpiSub" style="margin-top:8px;">Risk: {badge(risk_label, risk_color)}</div>
                  <div class="kpiSub" style="margin-top:10px;">{model_line}</div>
                  <div class="kpiSub" style="margin-top:6px;">Last update: {last_pred_time if last_pred_time else "—"}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric mini-cards (bigger, readable)
        m1, m2, m3 = st.columns([1.0, 1.0, 1.15], gap="small")

        with m1:
            st.markdown(gauge_html("Health", health, "0–100"), unsafe_allow_html=True)

        with m2:
            # Remaining time gauge: map hours to 0–100 for ring
            # Use a simple cap for visual: 0..250 hours -> 0..100
            capped = max(0.0, min(250.0, rul_hours))
            pct = (capped / 250.0) * 100.0
            st.markdown(
                f"""
                <div class="card">
                  <div class="kpiTitle">Remaining time</div>
                  <div class="gaugeWrap">
                    <div class="gauge" style="--deg:{int((pct/100)*360)}deg;">
                      <div class="gaugeInner">
                        <div class="gaugeVal">{int(round(rul_hours))}h</div>
                        <div class="gaugeLbl">estimate</div>
                      </div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with m3:
            # RPN indicator: show value + S/O/D if available
            if rpn_to_show is None:
                st.markdown(
                    """
                    <div class="card">
                      <div class="kpiTitle">RPN (risk priority)</div>
                      <div class="kpiSub">No RPN recorded for this component.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Map typical RPN 0..300 to gauge percent
                cap = max(0.0, min(300.0, float(rpn_to_show)))
                pct = (cap / 300.0) * 100.0
                breakdown = ""
                if sev is not None and occ is not None and det is not None and pd.notna(sev) and pd.notna(occ) and pd.notna(det):
                    breakdown = f"<div class='kpiSub' style='margin-top:8px;'>S {int(sev)} × O {int(occ)} × D {int(det)}</div>"

                st.markdown(
                    f"""
                    <div class="card">
                      <div class="kpiTitle">RPN (risk priority)</div>
                      <div class="gaugeWrap">
                        <div class="gauge" style="--deg:{int((pct/100)*360)}deg;">
                          <div class="gaugeInner">
                            <div class="gaugeVal">{int(round(float(rpn_to_show)))}</div>
                            <div class="gaugeLbl">0–300</div>
                          </div>
                        </div>
                        <div>
                          <div class="kpiSub">{badge(risk_label, risk_color)} (action level)</div>
                          {breakdown}
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Open alerts panel
        render_open_alerts(alerts_df)

        # Maintenance mode extra info (kept separate)
        if mode == "Maintenance / Engineer":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("<div class='kpiTitle'>Maintenance details</div>", unsafe_allow_html=True)

            # Confidence (shown only here)
            if conf is not None:
                st.markdown(f"<div class='kpiSub'>Prediction confidence: <b>{conf:.0%}</b></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='kpiSub'>Prediction confidence: —</div>", unsafe_allow_html=True)

            # Show the last few remaining-life points in a small table (audit-friendly)
            tail_series = series.tail(8).copy()
            if not tail_series.empty:
                tail_series["prediction_time"] = pd.to_datetime(tail_series["prediction_time"], errors="coerce")
                tail_series = tail_series.dropna(subset=["prediction_time"])
                tail_series = tail_series.sort_values("prediction_time", ascending=False)
                tail_series = tail_series.rename(
                    columns={
                        "prediction_time": "Time",
                        "predicted_value": "RUL (hrs)",
                        "confidence": "Confidence",
                    }
                )
                tail_series["Confidence"] = tail_series["Confidence"].apply(lambda x: f"{safe_float(x,0):.0%}")
                st.dataframe(tail_series, use_container_width=True, hide_index=True)

            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()


