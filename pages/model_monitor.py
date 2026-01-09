# pages/03_Model_Monitoring.py
import json
import pandas as pd
import streamlit as st
import altair as alt

from utils import load_df, validate_metrics, inject_global_styles, badge

alt.themes.enable("none")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Model Monitoring",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# SMALL UI HELPERS (no styling duplication)
# ----------------------------
def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def kpi(title: str, value: str, sub: str = ""):
    sub_html = f"<div class='kpiSub'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div class="card kpi">
          <div class="kpiTitle">{title}</div>
          <div class="kpiValue">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def safe_pct(x):
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return f"{v*100:.0f}%"
    except Exception:
        return None

def safe_dt(x):
    s = str(x).strip() if x is not None else ""
    return s if s else "—"

def get_models() -> pd.DataFrame:
    return load_df(
        """
        SELECT model_id, model_name, model_type, version, created_at, performance_metrics
        FROM predictive_models
        ORDER BY created_at DESC
        """
    )

def get_recent_predictions_for_model(model_id: int) -> pd.DataFrame:
    # Prediction history is what makes this page feel "real"
    # and helps USCIS see operational behavior.
    return load_df(
        """
        SELECT cp.prediction_time, cp.component_id, cp.prediction_type,
               cp.predicted_value, cp.confidence, cp.time_horizon
        FROM component_predictions cp
        WHERE cp.model_id = ?
        ORDER BY cp.prediction_time DESC
        LIMIT 400
        """,
        (model_id,),
    )

def get_component_label_map() -> pd.DataFrame:
    return load_df(
        """
        SELECT component_id, tail_number, type, name
        FROM components
        """
    )

def validation_badge(metrics_json: str):
    if metrics_json and validate_metrics(metrics_json):
        return badge("Validated", "#16A34A")
    return badge("Not validated", "#DC2626")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("### GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
inject_global_styles(dark_mode)

st.sidebar.caption("Model registry, monitoring, and export")

# ----------------------------
# HERO
# ----------------------------
st.markdown(
    """
    <div style="margin-bottom:14px;">
      <div style="font-size:2.05rem; font-weight:900; letter-spacing:-0.02em;">Model Monitoring</div>
      <div class="muted">Performance summary and operational traceability.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# LOAD MODELS
# ----------------------------
models = get_models()
if models.empty:
    card_open()
    st.markdown(badge("No models registered", "#2563EB"), unsafe_allow_html=True)
    st.markdown("<div class='kpiSub' style='margin-top:10px;'>No rows found in <b>predictive_models</b>.</div>", unsafe_allow_html=True)
    card_close()
    st.stop()

# Model selector (clean labels)
models = models.copy()
models["label"] = models.apply(
    lambda r: f"{r['model_name']} v{r['version']} ({r['model_type']}) • id {r['model_id']}",
    axis=1,
)
model_ids = models["model_id"].tolist()

card_open()
st.markdown("<div class='kpiTitle'>Model</div>", unsafe_allow_html=True)
selected_model_id = st.selectbox(
    " ",
    options=model_ids,
    format_func=lambda mid: models.loc[models["model_id"] == mid, "label"].iloc[0],
    label_visibility="collapsed",
)
card_close()

row = models[models["model_id"] == selected_model_id].iloc[0]
metrics_raw = row.get("performance_metrics", None) or ""
metrics_ok = validate_metrics(metrics_raw)
metrics = json.loads(metrics_raw) if metrics_ok else {}

# ----------------------------
# TOP KPI STRIP (decision ready)
# ----------------------------
pred_df = get_recent_predictions_for_model(int(selected_model_id))

# Coverage / behavior
last_run = safe_dt(pred_df["prediction_time"].iloc[0]) if not pred_df.empty else "—"
coverage = f"{pred_df['component_id'].nunique()}" if not pred_df.empty else "0"
conf_avg = None
if not pred_df.empty and "confidence" in pred_df.columns:
    c = pd.to_numeric(pred_df["confidence"], errors="coerce")
    if c.notna().any():
        conf_avg = f"{(c.mean()*100):.0f}%"

val_status = validation_badge(metrics_raw)

k1, k2, k3, k4 = st.columns(4, gap="large")
with k1:
    kpi("Model", f"{row['model_name']} v{row['version']}", f"{row['model_type']}")
with k2:
    kpi("Coverage", coverage, "Components with recent predictions")
with k3:
    kpi("Avg confidence", conf_avg if conf_avg else "—", "Recent predictions")
with k4:
    kpi("Validation", val_status, "Metrics JSON check")

# ----------------------------
# MODEL SUMMARY + EXPORT
# ----------------------------
left, right = st.columns([1.45, 1.0], gap="large")

with left:
    card_open()
    st.markdown("<div class='kpiTitle'>Summary</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-weight:850; margin-top:4px;'>{row['model_name']} v{row['version']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpiSub' style='margin-top:10px;'>Algorithm: <b>{row['model_type']}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpiSub' style='margin-top:6px;'>Created: <b>{safe_dt(row.get('created_at'))}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kpiSub' style='margin-top:6px;'>Last run: <b>{last_run}</b></div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

    colA, colB = st.columns([1, 1], gap="small")
    with colA:
        st.download_button(
            label="Download metrics (JSON)",
            data=json.dumps(metrics, indent=2) if metrics_ok else "{}",
            file_name=f"model_{selected_model_id}_metrics.json",
            mime="application/json",
        )
    with colB:
        st.download_button(
            label="Download recent predictions (CSV)",
            data=pred_df.to_csv(index=False).encode("utf-8") if not pred_df.empty else "prediction_time,component_id\n".encode("utf-8"),
            file_name=f"model_{selected_model_id}_recent_predictions.csv",
            mime="text/csv",
        )

    card_close()

with right:
    # Performance metrics – keep simple, readable
    precision = safe_pct(metrics.get("precision")) if metrics_ok else None
    recall = safe_pct(metrics.get("recall")) if metrics_ok else None
    accuracy = safe_pct(metrics.get("accuracy")) if metrics_ok else None
    f1 = safe_pct(metrics.get("f1_score")) if metrics_ok else None

    card_open()
    st.markdown("<div class='kpiTitle'>Performance</div>", unsafe_allow_html=True)
    if not metrics_ok:
        st.markdown("<div class='kpiSub'>Metrics JSON missing required fields.</div>", unsafe_allow_html=True)
        st.markdown("<div class='kpiSub'>Required: precision, recall, accuracy, f1_score.</div>", unsafe_allow_html=True)
    else:
        a, b = st.columns(2, gap="small")
        with a:
            kpi("Precision", precision or "—")
            kpi("Accuracy", accuracy or "—")
        with b:
            kpi("Recall", recall or "—")
            kpi("F1 score", f1 or "—")
    card_close()

# ----------------------------
# OPERATIONAL BEHAVIOR (charts)
# ----------------------------
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

cL, cR = st.columns([1.55, 1.0], gap="large")

with cL:
    card_open()
    st.markdown("<div class='kpiTitle'>Confidence over time</div>", unsafe_allow_html=True)

    if pred_df.empty:
        st.markdown("<div class='kpiSub'>No predictions found for this model.</div>", unsafe_allow_html=True)
    else:
        df = pred_df.copy()
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
        df = df.dropna(subset=["prediction_time"])

        # Use only remaining_life for clean signal, fallback to all
        if "prediction_type" in df.columns:
            only_rl = df[df["prediction_type"].astype(str).str.lower() == "remaining_life"]
            if not only_rl.empty:
                df = only_rl

        axis_color = "#A8B3C7" if dark_mode else "#334155"
        grid_op = 0.15 if dark_mode else 0.25

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("prediction_time:T", title="Time", axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
                y=alt.Y("confidence:Q", title="Confidence", axis=alt.Axis(labelColor=axis_color, titleColor=axis_color), scale=alt.Scale(domain=[0, 1])),
                tooltip=[
                    alt.Tooltip("prediction_time:T", title="Time"),
                    alt.Tooltip("component_id:Q", title="Component ID"),
                    alt.Tooltip("confidence:Q", title="Confidence", format=".0%"),
                ],
            )
            .properties(height=280)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridOpacity=grid_op)
        )
        st.altair_chart(chart, use_container_width=True)

    card_close()

with cR:
    card_open()
    st.markdown("<div class='kpiTitle'>Prediction mix</div>", unsafe_allow_html=True)

    if pred_df.empty or "prediction_type" not in pred_df.columns:
        st.markdown("<div class='kpiSub'>Not available.</div>", unsafe_allow_html=True)
    else:
        mix = pred_df["prediction_type"].astype(str).str.lower().value_counts().reset_index()
        mix.columns = ["type", "count"]

        axis_color = "#A8B3C7" if dark_mode else "#334155"
        grid_op = 0.15 if dark_mode else 0.25

        bar = (
            alt.Chart(mix)
            .mark_bar()
            .encode(
                x=alt.X("type:N", title="Type", axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
                y=alt.Y("count:Q", title="Count", axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
                tooltip=[alt.Tooltip("type:N", title="Type"), alt.Tooltip("count:Q", title="Count")],
            )
            .properties(height=280)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridOpacity=grid_op)
        )
        st.altair_chart(bar, use_container_width=True)

    card_close()

# ----------------------------
# RECENT PREDICTIONS (audit table)
# ----------------------------
card_open()
st.markdown("<div class='kpiTitle'>Recent predictions (audit)</div>", unsafe_allow_html=True)

if pred_df.empty:
    st.markdown("<div class='kpiSub'>No recent predictions found for this model.</div>", unsafe_allow_html=True)
else:
    comp_map = get_component_label_map()
    pred_show = pred_df.copy()

    # Join component labels when available
    if not comp_map.empty:
        pred_show = pred_show.merge(comp_map, on="component_id", how="left")
        pred_show["component"] = pred_show.apply(
            lambda r: f"{r.get('tail_number','—')} • {r.get('type','—')} • {r.get('name','—')} #{int(r['component_id'])}",
            axis=1,
        )
    else:
        pred_show["component"] = pred_show["component_id"].astype(str)

    keep_cols = ["prediction_time", "component", "prediction_type", "predicted_value", "confidence", "time_horizon"]
    keep_cols = [c for c in keep_cols if c in pred_show.columns]
    pred_show = pred_show[keep_cols].head(80)

    # Format
    if "confidence" in pred_show.columns:
        pred_show["confidence"] = pd.to_numeric(pred_show["confidence"], errors="coerce").apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—")
    if "predicted_value" in pred_show.columns:
        pred_show["predicted_value"] = pd.to_numeric(pred_show["predicted_value"], errors="coerce").apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    st.dataframe(pred_show, use_container_width=True, hide_index=True)

card_close()

# ----------------------------
# JSON VALIDATOR (kept, but cleaned)
# ----------------------------
card_open()
st.markdown("<div class='kpiTitle'>Validate metrics JSON</div>", unsafe_allow_html=True)
st.markdown("<div class='kpiSub'>Paste metrics JSON to confirm required fields.</div>", unsafe_allow_html=True)

user_json = st.text_area(" ", height=160, label_visibility="collapsed")

colX, colY = st.columns([1, 5])
with colX:
    validate_btn = st.button("Validate")

if validate_btn:
    if validate_metrics(user_json):
        st.success("Valid metrics JSON.")
    else:
        st.error("Missing required fields: precision, recall, accuracy, f1_score.")

card_close()
