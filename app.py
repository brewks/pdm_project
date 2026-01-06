import streamlit as st
import pandas as pd
import sqlite3
import json
import altair as alt
import os
from pathlib import Path  # ✅ ADDED (for reliable logo path)

# Ensure Altair doesn't override your styling (important)
alt.themes.enable("none")

# ----------------------------
# CONFIGURATION
# ----------------------------
DB_PATH = "ga_maintenance.db"
SQL_SEED_FILE = "full_pdm_seed.sql"


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
# GLOBAL STYLES (Design System)
# ----------------------------
def inject_global_styles(dark_mode: bool):
    # Airbus-ish palette: navy, slate, off-white, calm accent blue
    if dark_mode:
        bg = "#0B1220"            # deep navy (not black)
        bg2 = "#0E172A"           # slightly lighter navy for subtle gradient
        panel = "rgba(17, 27, 47, 0.86)"   # slate-navy cards
        border = "rgba(148, 163, 184, 0.14)"
        text = "rgba(226, 232, 240, 0.92)"   # not pure white
        muted = "rgba(226, 232, 240, 0.68)"
        shadow = "0 10px 28px rgba(0,0,0,0.35)"
        input_bg = "rgba(15, 23, 42, 0.75)"
        accent = "#5AA2FF"        # calm Airbus-like blue
        grid = "rgba(148, 163, 184, 0.10)"
    else:
        bg = "#F3F6FB"            # soft blue-gray (not white)
        bg2 = "#EEF3FA"
        panel = "rgba(255, 255, 255, 0.88)"  # slightly translucent, soft
        border = "rgba(15, 23, 42, 0.10)"
        text = "rgba(15, 23, 42, 0.92)"
        muted = "rgba(15, 23, 42, 0.65)"
        shadow = "0 10px 26px rgba(2, 6, 23, 0.08)"
        input_bg = "rgba(255, 255, 255, 0.92)"
        accent = "#1F6FEB"
        grid = "rgba(15, 23, 42, 0.08)"

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

          /* App background: subtle gradient (Airbus vibe) */
          .stApp {{
            background: radial-gradient(1200px 600px at 20% 0%, var(--bg2), var(--bg));
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
          }}

          .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2.0rem;
            max-width: 1200px;
          }}

          /* Hide Streamlit chrome */
          [data-testid="stToolbar"] {{ visibility: hidden; height: 0; }}
          footer {{ visibility: hidden; }}
          header {{ visibility: hidden; }}

          /* Typography: calmer + tighter */
          h1, h2, h3 {{ letter-spacing: -0.02em; }}
          .muted {{
            color: var(--muted);
            font-size: 0.96rem;
            line-height: 1.35rem;
          }}

          /* Card system */
          .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(8px);
          }}

          /* KPI styles */
          .kpiTitle {{
            font-size: 0.88rem;
            color: var(--muted);
            margin-bottom: 6px;
          }}
          .kpiValue {{
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--text);
          }}
          .kpiSub {{
            margin-top: 6px;
            color: var(--muted);
            font-size: 0.85rem;
          }}

          /* Badge */
          .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.80rem;
            font-weight: 650;
            color: white;
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

          /* Buttons: calm accent */
          .stButton > button {{
            border-radius: 12px;
            padding: 0.55rem 0.9rem;
            font-weight: 650;
            border: 1px solid var(--border);
            background: var(--accent);
            color: white;
          }}
          .stButton > button:hover {{
            filter: brightness(0.96);
          }}

          /* Tabs – reduce harsh colors */
          .stTabs [data-baseweb="tab"] {{
            color: var(--muted);
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
        "ok":  "#1F9D55",  # calmer green
        "warn":"#DFAF2C",  # muted amber
        "crit":"#D64545",  # softer red
        "info":"#3B82F6",  # calm blue
    }
    c = colors.get(level, "#64748B")
    return f'<span class="badge" style="background:{c};">{text}</span>'


def kpi_card(title: str, value: str, sub: str | None = None):
    sub_html = f'<div class="kpiSub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">{title}</div>
          <div class="kpiValue">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# AUTOMATIC DB RESTORATION IF MISSING
# ----------------------------
if not os.path.exists(DB_PATH):
    st.warning("Database file not found. Attempting to restore from SQL seed...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            with open(SQL_SEED_FILE, "r") as f:
                conn.executescript(f.read())
        st.success("Database successfully restored.")
    except Exception as e:
        st.error(f"Database restoration failed: {e}")


# ----------------------------
# DB HELPERS
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def load_df(query: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return pd.read_sql_query(query, conn)
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()


def validate_metrics(metrics_json: str) -> bool:
    try:
        data = json.loads(metrics_json)
        required = ["precision", "recall", "accuracy", "f1_score"]
        return all(key in data for key in required)
    except Exception:
        return False


# ----------------------------
# SIDEBAR (Keep it simple)
# ----------------------------
st.sidebar.markdown("## 🛩️ GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Keep sidebar minimal (product feel).")

# Apply styles after knowing dark mode
inject_global_styles(dark_mode)


# ----------------------------
# LOAD DATA
# ----------------------------
components_df = load_df(
    """
    SELECT component_id, tail_number, name, condition, remaining_useful_life, last_health_score
    FROM components
"""
)

predictions_df = load_df(
    """
    SELECT * FROM component_predictions
    ORDER BY prediction_time DESC
"""
)


# ----------------------------
# HERO (✅ UPDATED: title + logo at top-right)
# ----------------------------
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "logo.png"

h_left, h_right = st.columns([8, 1], vertical_alignment="center")

with h_left:
    st.markdown(
        """
        <div style="margin-bottom: 18px;">
          <h1 style="margin-bottom: 6px;">General Aviation Predictive Maintenance</h1>
          <div class="muted">
            Component-level remaining-useful-life (RUL) estimation and failure risk monitoring,
            optimized for GA maintenance workflows.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h_right:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
    else:
        st.caption("logo.png not found")


# ----------------------------
# SELECTOR (Hero control)
# ----------------------------
if components_df.empty:
    st.error("No aircraft components available. Database may not have loaded correctly.")
    st.stop()

component_names = [f"{row['tail_number']} — {row['name']}" for _, row in components_df.iterrows()]
component_map = {f"{row['tail_number']} — {row['name']}": row["component_id"] for _, row in components_df.iterrows()}

selector_card = st.container()
with selector_card:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Select a component")
    selected_component = st.selectbox("Aircraft component", component_names, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

comp_id = component_map[selected_component]
comp_data = components_df[components_df["component_id"] == comp_id].iloc[0]
comp_preds = predictions_df[predictions_df["component_id"] == comp_id]

# ----------------------------
# COMPUTE KPIs
# ----------------------------
avg_conf = float(comp_preds["confidence"].mean() * 100) if not comp_preds.empty else 0.0
avg_rul = float(comp_data["remaining_useful_life"]) if pd.notna(comp_data["remaining_useful_life"]) else 0.0
health_score = int(comp_data["last_health_score"]) if pd.notna(comp_data["last_health_score"]) else 0

crit = pd.DataFrame()
if not comp_preds.empty:
    crit = comp_preds[(comp_preds["prediction_type"] == "failure") & (comp_preds["confidence"] > 0.9)]
crit_count = int(crit.shape[0])

condition = str(comp_data["condition"]) if pd.notna(comp_data["condition"]) else "unknown"

# Status determination (simple)
if crit_count > 0:
    status_level = "crit"
    status_text = "Critical risk detected"
elif health_score < 60:
    status_level = "warn"
    status_text = "Degraded health"
else:
    status_level = "ok"
    status_text = "Normal"

# ----------------------------
# KPI STRIP (Consistent cards)
# ----------------------------
k1, k2, k3, k4 = st.columns(4, gap="large")

with k1:
    kpi_card("Remaining Useful Life", f"{avg_rul:.1f} h", f"Condition: {condition}")

with k2:
    kpi_card("Health Score", f"{health_score}", f"Status: {status_text}")

with k3:
    kpi_card("Avg Confidence", f"{avg_conf:.1f}%", "Across latest predictions")

with k4:
    kpi_card("Critical Alerts", f"{crit_count}", "Confidence > 90%")


# ----------------------------
# PRIMARY SECTION (Hero chart + status)
# ----------------------------
left, right = st.columns([2.2, 1], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### RUL trend")

    if not comp_preds.empty:
        alt_data = comp_preds[comp_preds["prediction_type"] == "remaining_life"].copy()
        if not alt_data.empty:
            # Ensure datetime
            alt_data["prediction_time"] = pd.to_datetime(alt_data["prediction_time"], errors="coerce")
            alt_data = alt_data.dropna(subset=["prediction_time"])

            chart = (
                alt.Chart(alt_data)
                .mark_line(point=True)
                .encode(
                    x=alt.X("prediction_time:T", title="Time"),
                    y=alt.Y("predicted_value:Q", title="RUL (hours)"),
                    tooltip=[
                        alt.Tooltip("prediction_time:T", title="Time"),
                        alt.Tooltip("predicted_value:Q", title="RUL"),
                        alt.Tooltip("confidence:Q", title="Confidence", format=".2f"),
                    ],
                )
                .properties(height=360)
                .configure_view(strokeOpacity=0)
                .configure_axis(
                    grid=True,
                    gridOpacity=0.15,
                    labelColor="#A8B3C7",
                    titleColor="#A8B3C7",
                    tickColor="rgba(0,0,0,0)",
                )
            )

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No remaining life predictions available.")
    else:
        st.info("No predictions available for this component yet.")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Component summary")
    st.markdown(f"**{selected_component}**")
    st.markdown(f"Tail/component condition: **{condition}**")
    st.markdown(f"Overall status: {badge(status_text, status_level)}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Alerts")

    if not crit.empty:
        row = crit.iloc[0]
        explanation = str(row.get("explanation", "") or "")
        explanation_preview = explanation if len(explanation) <= 180 else explanation[:180] + "…"
        st.markdown(badge("Critical", "crit") + " " + "**Failure predicted**", unsafe_allow_html=True)
        st.markdown(f"- Horizon: **{row.get('time_horizon', 'N/A')}**")
        st.markdown(f"- Confidence: **{float(row.get('confidence', 0))*100:.1f}%**")
        if explanation_preview.strip():
            st.markdown(f"- Notes: {explanation_preview}")
    else:
        st.markdown(badge("Normal", "ok") + " No critical alerts found.", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# DETAILS (Tabs)
# ----------------------------
tabs = st.tabs(["Trends", "Alerts", "Maintenance", "Model", "JSON Validator"])

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Recent predictions (latest 25)")
    if comp_preds.empty:
        st.info("No predictions available.")
    else:
        show = comp_preds.head(25).copy()
        cols = [c for c in ["prediction_time", "prediction_type", "predicted_value", "confidence", "time_horizon"] if c in show.columns]
        st.dataframe(show[cols], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Alerts")
    if comp_preds.empty:
        st.info("No alerts available.")
    else:
        failures = comp_preds[comp_preds["prediction_type"] == "failure"].copy()
        if failures.empty:
            st.markdown(badge("Normal", "ok") + " No failure predictions.", unsafe_allow_html=True)
        else:
            failures["prediction_time"] = pd.to_datetime(failures["prediction_time"], errors="coerce")
            failures = failures.sort_values("prediction_time", ascending=False).head(50)
            st.dataframe(
                failures[["prediction_time", "confidence", "time_horizon", "explanation"]],
                use_container_width=True,
                hide_index=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Maintenance (placeholder)")
    st.caption("If you already have preventive tasks page, link or render it here.")
    st.markdown("- Show due tasks\n- Recommend inspections/actions by RUL thresholds\n- Display last service events")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Model performance")

    precision = recall = accuracy = f1_score = "N/A"

    if not comp_preds.empty and "model_id" in comp_preds.columns:
        selected_model_id = comp_preds["model_id"].iloc[0]
        metrics_df = load_df(
            f"""
            SELECT performance_metrics FROM predictive_models
            WHERE model_id = {int(selected_model_id)}
            """
        )
        if not metrics_df.empty and metrics_df["performance_metrics"].iloc[0]:
            metrics_json = metrics_df["performance_metrics"].iloc[0]
            if validate_metrics(metrics_json):
                metrics = json.loads(metrics_json)
                precision = f"{metrics.get('precision', 0) * 100:.1f}%"
                recall = f"{metrics.get('recall', 0) * 100:.1f}%"
                accuracy = f"{metrics.get('accuracy', 0) * 100:.1f}%"
                f1_score = f"{metrics.get('f1_score', 0) * 100:.1f}%"
            else:
                st.warning("Invalid performance metrics JSON. Required: precision, recall, accuracy, f1_score")

    c1, c2, c3, c4 = st.columns(4, gap="large")
    with c1: kpi_card("Precision", precision)
    with c2: kpi_card("Recall", recall)
    with c3: kpi_card("Accuracy", accuracy)
    with c4: kpi_card("F1 Score", f1_score)

    st.markdown('</div>', unsafe_allow_html=True)

with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Test performance metrics JSON")
    st.caption("Paste JSON containing precision, recall, accuracy, f1_score and validate it.")

    input_metrics = st.text_area("Performance Metrics JSON", height=160, label_visibility="collapsed")

    colA, colB = st.columns([1, 5])
    with colA:
        validate_btn = st.button("Validate")

    if validate_btn:
        if validate_metrics(input_metrics):
            st.success("✅ Valid performance metrics JSON.")
        else:
            st.error("❌ Invalid JSON or missing required fields: precision, recall, accuracy, f1_score.")

    st.markdown('</div>', unsafe_allow_html=True)
