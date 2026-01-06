import os
import json
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt

# Keep Altair from forcing its own theme
alt.themes.enable("none")

# ----------------------------
# CONFIG
# ----------------------------
DB_PATH = "ga_maintenance.db"
SQL_SEED_FILE = "full_pdm_seed.sql"
LOGO_PATH = "logo.png"

st.set_page_config(
    page_title="GA Predictive Maintenance",
    page_icon="🛩️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# STYLES
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
            padding-top: 1.1rem;
            padding-bottom: 2.0rem;
            max-width: 92vw;
            padding-left: 2.25rem;
            padding-right: 2.25rem;
          }}

          [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: {sidebar_border};
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
            min-height: 108px;
          }}
          .kpiTitle {{
            font-size: 0.88rem;
            color: var(--muted);
            margin-bottom: 6px;
          }}
          .kpiValue {{
            font-size: 1.65rem;
            font-weight: 800;
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
            font-weight: 800;
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
          textarea.stTextArea textarea {{
            background: var(--input);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 12px;
            font-size: 14px;
          }}

          .stDataFrame, .stTable {{
            color: var(--text) !important;
          }}

          /* Buttons */
          .stButton > button {{
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            font-weight: 800;
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
            gap:14px;
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

          /* Make Streamlit columns feel less compressed */
          [data-testid="stHorizontalBlock"] {{
            gap: 1.25rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    return axis_label, grid_opacity


def badge(text: str, level: str) -> str:
    colors = {
        "ok": "#1F9D55",
        "warn": "#DFAF2C",
        "crit": "#D64545",
        "info": "#3B82F6",
        "muted": "#64748B",
    }
    c = colors.get(level, colors["muted"])
    return f'<span class="badge" style="background:{c};">{text}</span>'


def kpi_card(title: str, value: str, sub: str | None = None, badge_html: str | None = None):
    sub_html = f'<div class="kpiSub">{sub}</div>' if sub else ""
    badge_block = f"<div style='margin-top:10px;'>{badge_html}</div>" if badge_html else ""
    st.markdown(
        f"""
        <div class="card kpi">
          <div class="kpiTitle">{title}</div>
          <div class="kpiValue">{value}</div>
          {sub_html}
          {badge_block}
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_card(title: str, value_text: str, pct_0_100: float, sub: str | None = None, extra_html: str | None = None):
    pct = 0.0 if pd.isna(pct_0_100) else float(pct_0_100)
    pct = max(0.0, min(100.0, pct))
    deg = f"{pct * 3.6:.2f}deg"
    sub_html = f"<div class='kpiSub'>{sub}</div>" if sub else ""
    extra = f"<div style='margin-top:10px;'>{extra_html}</div>" if extra_html else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">{title}</div>
          <div class="gaugeWrap">
            <div class="gauge" style="--deg: {deg};">
              <div class="gaugeInner">
                <div class="gaugeVal">{value_text}</div>
                <div class="gaugeLbl">{int(round(pct))}%</div>
              </div>
            </div>
            <div style="flex:1;">
              {sub_html}
              {extra}
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clamp_int(x, lo, hi):
    try:
        return max(lo, min(hi, int(round(float(x)))))
    except Exception:
        return lo


def rpn_bucket(rpn: int) -> tuple[str, str]:
    # label, badge-level
    if rpn >= 700:
        return ("Critical", "crit")
    if rpn >= 450:
        return ("High", "warn")
    if rpn >= 200:
        return ("Medium", "info")
    return ("Low", "ok")


def compute_rpn(component_alerts: pd.DataFrame, component_preds: pd.DataFrame, health_score):
    # Severity (1–10) from worst open alert severity
    sev_score = 2
    if not component_alerts.empty and "severity" in component_alerts.columns:
        sev_order = {"critical": 9, "warning": 6, "advisory": 3}
        for s in ["critical", "warning", "advisory"]:
            if (component_alerts["severity"].astype(str).str.lower() == s).any():
                sev_score = sev_order[s]
                break

    # Occurrence (1–10) from max failure confidence
    occ_score = 2
    if not component_preds.empty and "prediction_type" in component_preds.columns:
        f = component_preds[component_preds["prediction_type"] == "failure"].copy()
        if not f.empty and "confidence" in f.columns:
            max_conf = pd.to_numeric(f["confidence"], errors="coerce").dropna()
            if not max_conf.empty:
                occ_score = clamp_int(max_conf.max() * 10, 1, 10)

    # Detection (1–10): lower health => higher detection score
    if pd.isna(health_score):
        det_score = 5
    else:
        det_score = clamp_int(10 - (float(health_score) / 10.0), 1, 10)

    rpn = int(sev_score * occ_score * det_score)  # 1..1000
    return rpn, sev_score, occ_score, det_score


def safe_dt(x) -> pd.Timestamp | None:
    try:
        t = pd.to_datetime(x, errors="coerce")
        if pd.isna(t):
            return None
        return t
    except Exception:
        return None


# ----------------------------
# DB RESTORE
# ----------------------------
if not os.path.exists(DB_PATH):
    st.warning("Database not found. Restoring from SQL seed...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            with open(SQL_SEED_FILE, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
        st.success("Database restored.")
    except Exception as e:
        st.error(f"Database restore failed: {e}")
        st.stop()


# ----------------------------
# DB HELPERS
# ----------------------------
@st.cache_data(show_spinner=False, ttl=45)
def load_df(query: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(query, conn)
    except Exception:
        # Keep UI clean; return empty and handle downstream
        return pd.DataFrame()


# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("## 🛩️ GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)

axis_label, grid_opacity = inject_global_styles(dark_mode)

st.sidebar.markdown("---")

tails = load_df("SELECT DISTINCT tail_number FROM aircraft ORDER BY tail_number;")
tail_list = []
if not tails.empty and "tail_number" in tails.columns:
    tail_list = tails["tail_number"].dropna().astype(str).tolist()

tail_filter = st.sidebar.selectbox("Aircraft (tail number)", options=["All"] + tail_list, index=0)


# ----------------------------
# HEADER (logo right)
# ----------------------------
h_left, h_right = st.columns([12, 2], vertical_alignment="center")
with h_left:
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
          <h1 style="margin:0;">General Aviation Predictive Maintenance</h1>
          <div class="muted">Clear health, risk, and maintenance signals for pilots and mechanics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with h_right:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)


# ----------------------------
# LOAD DATA
# ----------------------------
aircraft_df = load_df(
    """
    SELECT tail_number, model, manufacturer, total_hours, last_annual_date,
           predictive_status, last_predictive_analysis
    FROM aircraft
    ORDER BY tail_number
    """
)

components_df = load_df(
    """
    SELECT component_id, tail_number, name, type, condition,
           remaining_useful_life, last_health_score
    FROM components
    ORDER BY tail_number, type, name
    """
)

snapshot_df = load_df("SELECT * FROM dashboard_snapshot_view ORDER BY tail_number;")

alerts_df = load_df(
    """
    SELECT alert_id, tail_number, component_id, alert_type, severity, message,
           generated_time, resolved, confidence_score, notification_status
    FROM alerts
    WHERE resolved = 0
    ORDER BY generated_time DESC
    """
)

reco_df = load_df(
    """
    SELECT r.recommendation_id, r.component_id, r.timestamp, r.confidence, r.acknowledged, r.implemented,
           t.task_name, t.system, t.pilot_allowed, t.ac_43_ref
    FROM maintenance_recommendations r
    JOIN preventive_tasks t ON r.task_id = t.task_id
    ORDER BY r.timestamp DESC
    """
)

pred_df = load_df(
    """
    SELECT prediction_id, component_id, model_id, prediction_time, prediction_type,
           predicted_value, confidence, time_horizon, explanation
    FROM component_predictions
    ORDER BY prediction_time DESC
    """
)

if tail_filter != "All":
    if not aircraft_df.empty and "tail_number" in aircraft_df.columns:
        aircraft_df = aircraft_df[aircraft_df["tail_number"].astype(str) == str(tail_filter)]
    if not components_df.empty and "tail_number" in components_df.columns:
        components_df = components_df[components_df["tail_number"].astype(str) == str(tail_filter)]
    if not snapshot_df.empty and "tail_number" in snapshot_df.columns:
        snapshot_df = snapshot_df[snapshot_df["tail_number"].astype(str) == str(tail_filter)]
    if not alerts_df.empty and "tail_number" in alerts_df.columns:
        alerts_df = alerts_df[alerts_df["tail_number"].astype(str) == str(tail_filter)]
    if not pred_df.empty and "component_id" in pred_df.columns and not components_df.empty:
        pred_df = pred_df.merge(components_df[["component_id", "tail_number"]], on="component_id", how="inner")
    if not reco_df.empty and "component_id" in reco_df.columns and not components_df.empty:
        reco_df = reco_df.merge(components_df[["component_id", "tail_number"]], on="component_id", how="inner")

if aircraft_df.empty:
    st.error("No aircraft found.")
    st.stop()

if components_df.empty:
    st.error("No components found.")
    st.stop()


# ----------------------------
# TOP KPI STRIP (bigger + wider)
# ----------------------------
fleet_active_alerts = int(alerts_df.shape[0]) if alerts_df is not None else 0
fleet_crit = int((alerts_df["severity"].astype(str).str.lower() == "critical").sum()) if not alerts_df.empty and "severity" in alerts_df.columns else 0
fleet_warn = int((alerts_df["severity"].astype(str).str.lower() == "warning").sum()) if not alerts_df.empty and "severity" in alerts_df.columns else 0

avg_health = None
if not snapshot_df.empty and "health_score" in snapshot_df.columns:
    try:
        avg_health = float(pd.to_numeric(snapshot_df["health_score"], errors="coerce").dropna().mean())
    except Exception:
        avg_health = None

last_dates = aircraft_df["last_predictive_analysis"].dropna().tolist() if "last_predictive_analysis" in aircraft_df.columns else []
last_dt = max([safe_dt(x) for x in last_dates if safe_dt(x) is not None], default=None)

status_map = {
    "normal": ("Normal", "ok"),
    "monitoring": ("Monitor", "warn"),
    "attention_needed": ("Attention", "warn"),
    "maintenance_required": ("Maintenance", "crit"),
}
def fmt_status(s: str) -> tuple[str, str]:
    if not s:
        return ("Unknown", "muted")
    s = str(s).strip().lower()
    return status_map.get(s, ("Unknown", "muted"))

top_status = None
if "predictive_status" in aircraft_df.columns and not aircraft_df["predictive_status"].dropna().empty:
    top_status = aircraft_df["predictive_status"].dropna().astype(str).str.lower().value_counts().idxmax()

label, lvl = fmt_status(top_status) if top_status else ("Unknown", "muted")

kpi1, kpi2, kpi3, kpi4 = st.columns([1.35, 1.35, 1.15, 1.15], gap="large")
with kpi1:
    kpi_card("Active alerts", f"{fleet_active_alerts}", f"Critical: {fleet_crit}  •  Warning: {fleet_warn}")
with kpi2:
    # Keep gauge in strip (readable at a glance)
    if avg_health is None:
        kpi_card("Average health", "—", "0–100 scale")
    else:
        gauge_card("Average health", f"{int(round(avg_health))}", avg_health, "0–100 scale")
with kpi3:
    kpi_card("Last analysis", last_dt.strftime("%Y-%m-%d") if last_dt else "—", "Latest recorded run")
with kpi4:
    kpi_card("Overall status", label, None, badge_html=badge(label, lvl))

st.markdown("")


# ----------------------------
# COMPONENT SELECT (safe resolution, no KeyError)
# ----------------------------
comp_view = components_df.copy()
for col in ["tail_number", "type", "name"]:
    if col not in comp_view.columns:
        st.error("Components table is missing required columns.")
        st.stop()

comp_view["label"] = (
    comp_view["tail_number"].astype(str).fillna("—")
    + " • "
    + comp_view["type"].astype(str).fillna("component").str.replace("_", " ")
    + " • "
    + comp_view["name"].astype(str).fillna("—")
)

left, right = st.columns([2.2, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Component")
    selected_label = st.selectbox(
        "Component",
        options=comp_view["label"].tolist(),
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

sel_row = comp_view[comp_view["label"] == selected_label]
if sel_row.empty:
    st.error("Selection error. Please refresh.")
    st.stop()

selected_comp_id = int(sel_row["component_id"].iloc[0])
selected_tail = str(sel_row["tail_number"].iloc[0])
selected_type = str(sel_row["type"].iloc[0])
selected_name = str(sel_row["name"].iloc[0])

rul = sel_row["remaining_useful_life"].iloc[0] if "remaining_useful_life" in sel_row.columns else None
health = sel_row["last_health_score"].iloc[0] if "last_health_score" in sel_row.columns else None
condition = str(sel_row["condition"].iloc[0]) if "condition" in sel_row.columns and pd.notna(sel_row["condition"].iloc[0]) else "—"

comp_preds = pred_df[pred_df["component_id"] == selected_comp_id].copy() if not pred_df.empty and "component_id" in pred_df.columns else pd.DataFrame()
comp_alerts = alerts_df[alerts_df["component_id"] == selected_comp_id].copy() if not alerts_df.empty and "component_id" in alerts_df.columns else pd.DataFrame()
comp_reco = reco_df[reco_df["component_id"] == selected_comp_id].copy() if not reco_df.empty and "component_id" in reco_df.columns else pd.DataFrame()

# Latest RUL prediction
latest_rul_pred = None
latest_rul_conf = None
latest_rul_time = None
if not comp_preds.empty and "prediction_type" in comp_preds.columns:
    r = comp_preds[comp_preds["prediction_type"] == "remaining_life"].copy()
    if not r.empty:
        r["prediction_time"] = pd.to_datetime(r["prediction_time"], errors="coerce")
        r = r.dropna(subset=["prediction_time"]).sort_values("prediction_time", ascending=False)
        if not r.empty:
            latest_rul_pred = float(pd.to_numeric(r["predicted_value"].iloc[0], errors="coerce"))
            latest_rul_conf = float(pd.to_numeric(r["confidence"].iloc[0], errors="coerce"))
            latest_rul_time = r["prediction_time"].iloc[0]

# Simple risk label
if not comp_alerts.empty and "severity" in comp_alerts.columns and (comp_alerts["severity"].astype(str).str.lower() == "critical").any():
    risk_lbl, risk_lvl = ("High", "crit")
elif not comp_alerts.empty and "severity" in comp_alerts.columns and (comp_alerts["severity"].astype(str).str.lower() == "warning").any():
    risk_lbl, risk_lvl = ("Medium", "warn")
else:
    risk_lbl, risk_lvl = ("Low", "ok")

# RPN (derived)
rpn, rpn_sev, rpn_occ, rpn_det = compute_rpn(comp_alerts, comp_preds, health)
rpn_label, rpn_level = rpn_bucket(rpn)
rpn_pct = max(0.0, min(100.0, (rpn / 1000.0) * 100.0))


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Summary")
    st.markdown(f"**{selected_tail}** • {selected_type.replace('_',' ')} • **{selected_name}**")
    st.markdown(f"Condition: **{condition.replace('_',' ')}**")
    st.markdown(f"Risk: {badge(risk_lbl, risk_lvl)}", unsafe_allow_html=True)
    st.markdown("---")

    a1, a2, a3 = st.columns(3, gap="large")

    with a1:
        if pd.isna(health):
            gauge_card("Health", "—", 0, "Higher is better")
        else:
            gauge_card("Health", f"{int(float(health))}", float(health), "Higher is better")

    with a2:
        disp_rul = latest_rul_pred if latest_rul_pred is not None else (float(rul) if rul is not None and pd.notna(rul) else None)
        if disp_rul is None:
            gauge_card("Remaining time", "—", 0, "Hours (estimate)")
        else:
            # Gauge normalization for quick glance: 0–500h -> 0–100%
            pct = max(0.0, min(100.0, (float(disp_rul) / 500.0) * 100.0))
            gauge_card("Remaining time", f"{float(disp_rul):.0f}h", pct, "Hours (estimate)")

    with a3:
        gauge_card(
            "RPN (risk priority)",
            f"{rpn}",
            rpn_pct,
            "0–1000 scale",
            extra_html=f"{badge(rpn_label, rpn_level)}<div class='muted' style='margin-top:6px;'>Sev {rpn_sev} × Lik {rpn_occ} × Det {rpn_det}</div>",
        )

    if latest_rul_time is not None and latest_rul_conf is not None:
        st.caption(f"Last RUL update: {latest_rul_time.strftime('%Y-%m-%d %H:%M')} • Confidence: {latest_rul_conf*100:.0f}%")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")


# ----------------------------
# CHART + ALERTS
# ----------------------------
c1, c2 = st.columns([2.25, 1], gap="large")

with c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### RUL trend")

    if comp_preds.empty:
        st.info("No prediction history available for this component.")
    else:
        trend = comp_preds[comp_preds["prediction_type"] == "remaining_life"].copy() if "prediction_type" in comp_preds.columns else pd.DataFrame()
        if trend.empty:
            st.info("No remaining-life history available for this component.")
        else:
            trend["prediction_time"] = pd.to_datetime(trend["prediction_time"], errors="coerce")
            trend = trend.dropna(subset=["prediction_time"])
            trend = trend.sort_values("prediction_time")

            chart = (
                alt.Chart(trend)
                .mark_line(point=True)
                .encode(
                    x=alt.X("prediction_time:T", title="Time"),
                    y=alt.Y("predicted_value:Q", title="Hours remaining"),
                    tooltip=[
                        alt.Tooltip("prediction_time:T", title="Time"),
                        alt.Tooltip("predicted_value:Q", title="Hours"),
                        alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                        alt.Tooltip("time_horizon:N", title="Horizon"),
                    ],
                )
                .properties(height=360)
                .configure_view(strokeOpacity=0)
                .configure_axis(
                    grid=True,
                    gridOpacity=grid_opacity,
                    labelColor=axis_label,
                    titleColor=axis_label,
                    tickColor="rgba(0,0,0,0)",
                )
            )
            st.altair_chart(chart, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Open alerts")

    if comp_alerts.empty:
        st.markdown(badge("None", "ok") + " No open alerts for this component.", unsafe_allow_html=True)
    else:
        show = comp_alerts.copy()
        if "generated_time" in show.columns:
            show["generated_time"] = pd.to_datetime(show["generated_time"], errors="coerce")
            show = show.sort_values("generated_time", ascending=False)
        show = show.head(6)

        for _, row in show.iterrows():
            sev = str(row.get("severity", "")).lower()
            lvl = "crit" if sev == "critical" else ("warn" if sev == "warning" else "info")
            msg = str(row.get("message", "")).strip()
            ts = row.get("generated_time")
            ts_txt = ts.strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "—"

            st.markdown(f"{badge(sev.capitalize() if sev else 'Alert', lvl)} <span class='muted'>({ts_txt})</span>", unsafe_allow_html=True)
            st.write(msg if msg else "—")

            cs = row.get("confidence_score")
            if pd.notna(cs):
                st.caption(f"Confidence: {float(cs)*100:.0f}%")

            st.markdown("---")

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# NEXT STEPS
# ----------------------------
st.markdown("")
t1, t2 = st.columns([1.6, 1.1], gap="large")

with t1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recommended actions")

    if comp_reco.empty:
        st.info("No recommendations recorded for this component.")
    else:
        view = comp_reco.copy()
        if "timestamp" in view.columns:
            view["timestamp"] = pd.to_datetime(view["timestamp"], errors="coerce")
            view = view.sort_values("timestamp", ascending=False)
        view = view.head(10)

        cols = ["timestamp", "task_name", "system", "pilot_allowed", "ac_43_ref", "confidence", "acknowledged", "implemented"]
        cols = [c for c in cols if c in view.columns]
        st.dataframe(view[cols], use_container_width=True, hide_index=True)

        st.download_button(
            "Download (CSV)",
            data=view[cols].to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_tail}_{selected_name}_recommendations.csv".replace(" ", "_"),
            mime="text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)

with t2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Latest predictions")

    if comp_preds.empty:
        st.info("No predictions available.")
    else:
        p = comp_preds.copy()
        if "prediction_time" in p.columns:
            p["prediction_time"] = pd.to_datetime(p["prediction_time"], errors="coerce")
            p = p.sort_values("prediction_time", ascending=False)
        p = p.head(12)

        show_cols = ["prediction_time", "prediction_type", "predicted_value", "confidence", "time_horizon"]
        show_cols = [c for c in show_cols if c in p.columns]
        st.dataframe(p[show_cols], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
