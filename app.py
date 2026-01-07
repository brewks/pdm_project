import os
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt


# ----------------------------
# CONFIG
# ----------------------------
APP_TITLE = "General Aviation Predictive Maintenance"
DB_PATH = os.getenv("DB_PATH", "ga_maintenance.db")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✈️",
    layout="wide",
)


# ----------------------------
# STYLES (your function + targeted fixes)
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
        sidebar_text = "rgba(226, 232, 240, 0.92)"
        sidebar_muted = "rgba(226, 232, 240, 0.70)"
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
        # Fix: sidebar labels disappearing in light mode
        sidebar_text = "rgba(15, 23, 42, 0.92)"
        sidebar_muted = "rgba(15, 23, 42, 0.70)"

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
            padding-bottom: 1.4rem;
            max-width: 96vw;
            padding-left: 1.6rem;
            padding-right: 1.6rem;
          }}

          /* Reduce vertical whitespace between sections */
          .element-container {{ margin-bottom: 0.55rem; }}
          div[data-testid="stVerticalBlock"] > div {{ gap: 0.75rem; }}

          [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          section[data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: {sidebar_border};
          }}

          /* Fix sidebar text in light mode (and keep consistent in dark) */
          section[data-testid="stSidebar"] * {{
            color: {sidebar_text} !important;
          }}
          section[data-testid="stSidebar"] .stCaption {{
            color: {sidebar_muted} !important;
          }}

          h1, h2, h3 {{ letter-spacing: -0.02em; margin-bottom: 0.25rem; }}
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

          /* KPI cards: bigger + strip feel */
          .card.kpi {{
            padding: 18px 20px;
            min-height: 110px;
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
            line-height: 1.2;
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

          /* Make columns feel less compressed */
          [data-testid="stHorizontalBlock"] {{
            gap: 1.0rem;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# DATA ACCESS
# ----------------------------
@st.cache_data(show_spinner=False)
def read_sql(query: str, params: Tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def fmt_dt(x) -> str:
    if x is None or str(x).strip() == "":
        return "—"
    s = str(x)
    # SQLite datetime('now','localtime') usually returns "YYYY-MM-DD HH:MM:SS"
    return s


# ----------------------------
# UI HELPERS
# ----------------------------
def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color};">{text}</span>'


def risk_from_rul(rul_hours: Optional[float]) -> Tuple[str, str]:
    """
    Simple, decision-friendly mapping.
    (You can tune these thresholds later.)
    """
    if rul_hours is None:
        return "Unknown", "#64748B"
    if rul_hours < 25:
        return "High", "#DC2626"
    if rul_hours < 75:
        return "Medium", "#2563EB"
    return "Low", "#16A34A"


def rpn_level(rpn: Optional[float]) -> Tuple[str, str]:
    if rpn is None:
        return "Unknown", "#64748B"
    rpn = float(rpn)
    # Common, interpretable bands (tunable)
    if rpn >= 300:
        return "High", "#DC2626"
    if rpn >= 120:
        return "Medium", "#2563EB"
    return "Low", "#16A34A"


def gauge(value: float, label: str, suffix: str = "", max_value: float = 100.0) -> str:
    value = max(0.0, min(float(value), float(max_value)))
    deg = (value / max_value) * 360.0
    main = f"{int(round(value))}{suffix}".strip()
    return f"""
    <div class="gaugeWrap">
      <div class="gauge" style="--deg:{deg}deg;">
        <div class="gaugeInner">
          <div class="gaugeVal">{main}</div>
          <div class="gaugeLbl">{label}</div>
        </div>
      </div>
    </div>
    """


def kpi_card(title: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpiSub">{sub}</div>' if sub else ""
    return f"""
    <div class="card kpi">
      <div class="kpiTitle">{title}</div>
      <div class="kpiValue">{value}</div>
      {sub_html}
    </div>
    """


# ----------------------------
# APP
# ----------------------------
def main():
    # Sidebar (clean, professional)
    st.sidebar.markdown("### GA PdM")
    dark_mode = st.sidebar.toggle("Dark mode", value=True)

    inject_global_styles(dark_mode)

    mode = st.sidebar.radio(
        "View mode",
        ["Pilot / Operator", "Maintenance / Engineer"],
        index=0,
        help="Pilot mode is simplified. Maintenance mode includes technical detail (RPN, confidence, model info).",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Data source: SQLite (ga_maintenance.db)")

    # Filters
    snap = read_sql("SELECT * FROM dashboard_snapshot_view ORDER BY tail_number;")
    tails = ["All"] + sorted(snap["tail_number"].dropna().unique().tolist()) if not snap.empty else ["All"]
    selected_tail = st.sidebar.selectbox("Aircraft (tail number)", tails, index=0)

    # Header
    h1, h2 = st.columns([5, 1.2])
    with h1:
        st.markdown(f"## {APP_TITLE}")
        st.markdown('<div class="muted">Clear health, risk, and maintenance signals for pilots and mechanics.</div>', unsafe_allow_html=True)
    with h2:
        # Optional: if you have a logo file, keep it. If not, nothing breaks.
        for logo in ("logo.png", "logo_white.png", "assets/logo.png"):
            if os.path.exists(logo):
                st.image(logo, use_container_width=True)
                break

    # Fleet KPI strip (wide)
    # Fleet aggregation uses dashboard_snapshot_view
    if selected_tail == "All":
        snap_f = snap.copy()
    else:
        snap_f = snap[snap["tail_number"] == selected_tail].copy()

    total_alerts = int(safe_float(snap_f["active_alerts"].sum() if not snap_f.empty else 0, 0))
    avg_health = safe_float(snap_f["health_score"].mean() if not snap_f.empty else None, 0.0)
    last_analysis = snap_f["last_prediction_time"].dropna().max() if not snap_f.empty else None

    # Fleet status from aircraft.predictive_status (joined via snapshot view)
    # snap has predictive_status; pick "worst" if multiple
    status_rank = {"maintenance_required": 4, "attention_needed": 3, "monitoring": 2, "normal": 1}
    fleet_status = "normal"
    if not snap_f.empty and "predictive_status" in snap_f.columns:
        vals = [v for v in snap_f["predictive_status"].dropna().tolist()]
        if vals:
            fleet_status = sorted(vals, key=lambda x: status_rank.get(str(x), 0), reverse=True)[0]

    status_label = fleet_status.replace("_", " ").title()
    status_color = {"Normal": "#16A34A", "Monitoring": "#2563EB", "Attention Needed": "#F59E0B", "Maintenance Required": "#DC2626"}.get(status_label, "#64748B")

    k1, k2, k3, k4 = st.columns([1.25, 1.25, 1.25, 1.0])
    with k1:
        st.markdown(kpi_card("Active alerts", f"{total_alerts}", "Unresolved across selected fleet"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_card("Average health", f"{int(round(avg_health))}/100", "Higher is better"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_card("Last analysis", fmt_dt(last_analysis), "Most recent prediction timestamp"), unsafe_allow_html=True)
    with k4:
        st.markdown(
            f"""
            <div class="card kpi">
              <div class="kpiTitle">Overall status</div>
              <div class="kpiValue">{status_label}</div>
              <div class="kpiSub">{badge(status_label, status_color)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Load components
    comps = read_sql(
        """
        SELECT component_id, tail_number, name, type, condition,
               remaining_useful_life, last_health_score
        FROM components
        ORDER BY tail_number, type, name;
        """
    )

    if selected_tail != "All" and not comps.empty:
        comps = comps[comps["tail_number"] == selected_tail].copy()

    st.markdown("")  # small spacer (kept minimal)

    # Component selector
    left, right = st.columns([2.2, 1.0])

    with left:
        st.markdown("### Component")
        if comps.empty:
            st.warning("No components found for this selection.")
            return

        # SAFE selection: store IDs and use format_func, no label->id dict required
        comp_ids = comps["component_id"].astype(int).tolist()

        def comp_label(cid: int) -> str:
            row = comps[comps["component_id"] == cid].iloc[0]
            tail = row["tail_number"]
            typ = str(row["type"]).replace("_", " ").title()
            nm = row["name"]
            return f"{tail} • {typ} • {nm}  (#{cid})"

        selected_comp_id = st.selectbox(
            "Select component",
            comp_ids,
            format_func=comp_label,
            index=0,
        )

    # Pull latest per-component prediction bundle with RPN
    # RPN comes from component_predictions_with_rpn (rpn_calc / factors)
    preds = read_sql(
        """
        SELECT *
        FROM component_predictions_with_rpn
        WHERE component_id = ?
        ORDER BY prediction_time DESC
        LIMIT 250;
        """,
        (int(selected_comp_id),),
    )

    # Component basics
    comp_row = comps[comps["component_id"] == int(selected_comp_id)].iloc[0]
    rul = comp_row["remaining_useful_life"]
    health = comp_row["last_health_score"]
    condition = comp_row["condition"]
    tail_number = comp_row["tail_number"]
    comp_type = comp_row["type"]
    comp_name = comp_row["name"]

    risk_lbl, risk_col = risk_from_rul(None if pd.isna(rul) else float(rul))

    # Latest RUL prediction confidence (if present)
    rul_conf = None
    if not preds.empty:
        pr = preds[preds["prediction_type"] == "remaining_life"]
        if not pr.empty:
            rul_conf = pr.iloc[0]["confidence"]

    # Latest RPN (if present)
    latest_rpn = None
    s = o = d = None
    failure_mode = None
    if not preds.empty:
        # Use calculated if stored is null
        latest = preds.iloc[0]
        latest_rpn = latest.get("rpn_calc")
        if pd.isna(latest_rpn):
            latest_rpn = latest.get("rpn")
        failure_mode = latest.get("failure_mode")
        s = latest.get("fmea_severity")
        o = latest.get("fmea_occurrence_base")
        d = latest.get("fmea_detection_base")

    rpn_lbl, rpn_col = rpn_level(None if (latest_rpn is None or pd.isna(latest_rpn)) else float(latest_rpn))

    # Alerts (open)
    open_alerts = read_sql(
        """
        SELECT severity, alert_type, message, generated_time
        FROM alerts
        WHERE component_id = ? AND resolved = 0
        ORDER BY generated_time DESC
        LIMIT 50;
        """,
        (int(selected_comp_id),),
    )
    crit_count = int((open_alerts["severity"] == "critical").sum()) if not open_alerts.empty else 0
    warn_count = int((open_alerts["severity"] == "warning").sum()) if not open_alerts.empty else 0

    # Summary + gauges
    with right:
        st.markdown("### Summary")
        st.markdown(
            f"""
            <div class="card">
              <div style="font-weight:800; margin-bottom:6px;">{tail_number} • {str(comp_type).replace("_"," ").title()} • {comp_name}</div>
              <div class="muted" style="margin-top:2px;">Condition: <b>{condition}</b></div>
              <div class="muted" style="margin-top:2px;">Risk: {badge(risk_lbl, risk_col)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Row of three compact KPI gauges (no HTML artifacts)
        g1, g2, g3 = st.columns([1, 1, 1])

        with g1:
            st.markdown(
                f"""
                <div class="card">
                  <div class="kpiTitle">Health</div>
                  {gauge(0 if pd.isna(health) else float(health), "score", "", 100)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with g2:
            # RUL displayed in hours
            rul_val = 0.0 if (rul is None or pd.isna(rul)) else float(rul)
            # Cap gauge visually at 300h (tunable)
            st.markdown(
                f"""
                <div class="card">
                  <div class="kpiTitle">Remaining time</div>
                  {gauge(rul_val, "hours", "h", 300)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with g3:
            # RPN: show only in maintenance mode, otherwise show Confidence (simpler)
            if mode == "Maintenance / Engineer":
                rpn_val = 0.0 if (latest_rpn is None or pd.isna(latest_rpn)) else float(latest_rpn)
                # RPN gauge scale: 0–1000 (10*10*10)
                st.markdown(
                    f"""
                    <div class="card">
                      <div class="kpiTitle">RPN</div>
                      {gauge(rpn_val, "priority", "", 1000)}
                      <div class="kpiSub">{badge(rpn_lbl, rpn_col)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                conf_pct = 0 if (rul_conf is None or pd.isna(rul_conf)) else int(round(float(rul_conf) * 100))
                st.markdown(
                    f"""
                    <div class="card">
                      <div class="kpiTitle">Prediction confidence</div>
                      {gauge(conf_pct, "confidence", "%", 100)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Maintenance mode: show S×O×D breakdown (clean, not verbose)
        if mode == "Maintenance / Engineer":
            if s is not None and not pd.isna(s) and o is not None and not pd.isna(o) and d is not None and not pd.isna(d):
                st.markdown(
                    f"""
                    <div class="card" style="margin-top:10px;">
                      <div class="kpiTitle">RPN breakdown</div>
                      <div style="display:flex; gap:10px; flex-wrap:wrap;">
                        <div><b>Severity</b>: {int(s)}</div>
                        <div><b>Occurrence</b>: {int(o)}</div>
                        <div><b>Detection</b>: {int(d)}</div>
                      </div>
                      <div class="kpiSub">Failure mode: <b>{failure_mode if failure_mode else "—"}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="card" style="margin-top:10px;">
                      <div class="kpiTitle">RPN breakdown</div>
                      <div class="muted">No FMEA ratings found for this component/failure mode.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Open alerts summary (simple + actionable)
        if open_alerts.empty:
            st.markdown(
                f"""
                <div class="card" style="margin-top:10px;">
                  <div class="kpiTitle">Open alerts</div>
                  <div class="kpiSub">{badge("None", "#16A34A")} No open alerts for this component.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            sev_color = {"critical": "#DC2626", "warning": "#2563EB", "advisory": "#64748B"}
            st.markdown(
                f"""
                <div class="card" style="margin-top:10px;">
                  <div class="kpiTitle">Open alerts</div>
                  <div class="kpiSub">
                    {badge(f"Critical: {crit_count}", sev_color["critical"])}&nbsp;
                    {badge(f"Warning: {warn_count}", sev_color["warning"])}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----------------------------
    # RUL trend (main chart)
    # ----------------------------
    st.markdown("### RUL trend")

    rul_series = pd.DataFrame()
    if not preds.empty:
        rul_series = preds[preds["prediction_type"] == "remaining_life"].copy()

    if rul_series.empty:
        st.markdown(
            '<div class="card"><div class="muted">No RUL time-series available for this component.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        rul_series["prediction_time"] = pd.to_datetime(rul_series["prediction_time"], errors="coerce")
        rul_series = rul_series.dropna(subset=["prediction_time"])
        rul_series = rul_series.sort_values("prediction_time")

        # Pilot mode: simple line
        base = alt.Chart(rul_series).encode(
            x=alt.X("prediction_time:T", title="Time"),
            y=alt.Y("predicted_value:Q", title="Remaining time (hours)"),
            tooltip=[
                alt.Tooltip("prediction_time:T", title="Time"),
                alt.Tooltip("predicted_value:Q", title="RUL (hrs)", format=".0f"),
                alt.Tooltip("confidence:Q", title="Confidence", format=".0%"),
            ],
        )

        line = base.mark_line(point=True)

        chart = (
            (line)
            .properties(height=280)
            .configure_axis(
                labelColor="#A8B3C7" if dark_mode else "#334155",
                titleColor="#A8B3C7" if dark_mode else "#334155",
                gridColor="rgba(148,163,184,0.12)" if dark_mode else "rgba(15,23,42,0.10)",
                gridOpacity=0.15 if dark_mode else 0.25,
            )
            .configure_view(strokeOpacity=0)
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.altair_chart(chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------------
    # Pilot vs Maintenance sections
    # ----------------------------
    if mode == "Pilot / Operator":
        st.markdown("### What to do")
        msg = "Continue normal operations."
        if risk_lbl == "Medium":
            msg = "Plan maintenance soon. Monitor this component."
        if risk_lbl == "High":
            msg = "Maintenance attention recommended before extended operations."
        st.markdown(
            f"""
            <div class="card">
              <div style="font-weight:900; font-size:1.05rem; margin-bottom:6px;">Recommendation</div>
              <div class="muted">{msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show only the most recent 5 open alerts (if any)
        if not open_alerts.empty:
            st.markdown("### Latest alerts")
            show = open_alerts.head(5).copy()
            show["generated_time"] = show["generated_time"].apply(fmt_dt)
            st.dataframe(show, use_container_width=True, hide_index=True)

    else:
        st.markdown("### Maintenance details")

        # Latest model info (minimal)
        model_info = read_sql(
            """
            SELECT pm.model_name, pm.version, pm.model_type, pm.training_date, pm.deployment_date
            FROM predictive_models pm
            JOIN component_predictions cp ON pm.model_id = cp.model_id
            WHERE cp.component_id = ?
            ORDER BY cp.prediction_time DESC
            LIMIT 1;
            """,
            (int(selected_comp_id),),
        )

        mcol1, mcol2 = st.columns([1.2, 1.0])

        with mcol1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Latest model used**")
            if model_info.empty:
                st.markdown('<div class="muted">No model record found for this component.</div>', unsafe_allow_html=True)
            else:
                row = model_info.iloc[0].to_dict()
                st.markdown(
                    f"""
                    <div class="muted"><b>{row.get("model_name","—")}</b> • v{row.get("version","—")} • {row.get("model_type","—")}</div>
                    <div class="muted">Trained: {fmt_dt(row.get("training_date"))} • Deployed: {fmt_dt(row.get("deployment_date"))}</div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        with mcol2:
            # Confidence as its own decision-support gauge (kept in maintenance mode too)
            conf_pct = 0 if (rul_conf is None or pd.isna(rul_conf)) else int(round(float(rul_conf) * 100))
            st.markdown(
                f"""
                <div class="card">
                  <div class="kpiTitle">RUL confidence</div>
                  {gauge(conf_pct, "confidence", "%", 100)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show open alerts table fully
        st.markdown("### Open alerts (unresolved)")
        if open_alerts.empty:
            st.info("No open alerts for this component.")
        else:
            df = open_alerts.copy()
            df["generated_time"] = df["generated_time"].apply(fmt_dt)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Optional: show latest predictions table (trimmed)
        st.markdown("### Latest predictions")
        if preds.empty:
            st.info("No predictions found.")
        else:
            cols = [
                "prediction_time",
                "prediction_type",
                "predicted_value",
                "confidence",
                "time_horizon",
                "failure_mode",
                "rpn_calc",
            ]
            slim = preds[[c for c in cols if c in preds.columns]].copy()
            slim["prediction_time"] = slim["prediction_time"].apply(fmt_dt)
            st.dataframe(slim.head(25), use_container_width=True, hide_index=True)

    # Footer note (short, non-fluffy)
    st.caption("Operational view. Values depend on latest recorded predictions and unresolved alerts.")


if __name__ == "__main__":
    main()
