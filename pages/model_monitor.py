import json
import pandas as pd
import streamlit as st
import altair as alt

from utils import load_df, validate_metrics, inject_global_styles, badge, kpi_card, altair_axis_colors

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
# DATA
# ----------------------------
def get_models() -> pd.DataFrame:
    return load_df(
        """
        SELECT model_id, model_name, model_type, version, created_at, performance_metrics
        FROM predictive_models
        ORDER BY created_at DESC
        """
    )

def get_recent_predictions(model_id: int) -> pd.DataFrame:
    # Operational trace = production-grade
    return load_df(
        """
        SELECT prediction_time, component_id, prediction_type,
               predicted_value, confidence, time_horizon
        FROM component_predictions
        WHERE model_id = ?
        ORDER BY prediction_time DESC
        LIMIT 500
        """,
        (model_id,),
    )

def get_components_map() -> pd.DataFrame:
    return load_df(
        """
        SELECT component_id, tail_number, type, name
        FROM components
        """
    )

def fmt_dt(x) -> str:
    s = str(x).strip() if x is not None else ""
    return s if s else "—"

def pct_or_dash(x):
    try:
        if x is None:
            return "—"
        v = float(x)
        if pd.isna(v):
            return "—"
        return f"{v*100:.0f}%"
    except Exception:
        return "—"

def validation_chip(metrics_json: str) -> str:
    if metrics_json and validate_metrics(metrics_json):
        return badge("Validated", "#16A34A")
    return badge("Not validated", "#DC2626")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.markdown("### GA PdM")
dark_mode = st.sidebar.toggle("Dark mode", value=True)
inject_global_styles(dark_mode)
st.sidebar.caption("Model registry, performance, and traceability")

# ----------------------------
# HERO
# ----------------------------
st.markdown(
    """
    <div style="margin-bottom:14px;">
      <div style="font-size:2.05rem; font-weight:900; letter-spacing:-0.02em;">Model Monitoring</div>
      <div class="muted">Registry, performance, and operational trace.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

models = get_models()
if models.empty:
    st.markdown(
        f"""
        <div class="card">
          {badge("No models found", "#2563EB")}
          <div class="kpiSub" style="margin-top:10px;">
            No rows available in <b>predictive_models</b>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Clean selector label
models = models.copy()
models["label"] = models.apply(
    lambda r: f"{r['model_name']} v{r['version']} ({r['model_type']}) • id {r['model_id']}",
    axis=1,
)
model_ids = models["model_id"].tolist()

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='kpiTitle'>Model</div>", unsafe_allow_html=True)
selected_model_id = st.selectbox(
    " ",
    options=model_ids,
    format_func=lambda mid: models.loc[models["model_id"] == mid, "label"].iloc[0],
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

row = models[models["model_id"] == selected_model_id].iloc[0]
metrics_raw = row.get("performance_metrics", "") or ""
metrics_ok = validate_metrics(metrics_raw)
metrics = json.loads(metrics_raw) if metrics_ok else {}

pred = get_recent_predictions(int(selected_model_id))
last_run = fmt_dt(pred["prediction_time"].iloc[0]) if not pred.empty else "—"

# Coverage
coverage = str(pred["component_id"].nunique()) if not pred.empty else "0"

# Avg confidence
avg_conf = "—"
if not pred.empty and "confidence" in pred.columns:
    c = pd.to_numeric(pred["confidence"], errors="coerce")
    if c.notna().any():
        avg_conf = f"{(c.mean()*100):.0f}%"

# Validation chip
val_chip = validation_chip(metrics_raw)

# ----------------------------
# TOP KPI STRIP
# ----------------------------
c1, c2, c3, c4 = st.columns([1.35, 1.05, 1.05, 1.10], gap="large")

with c1:
    st.markdown(kpi_card("Active model", f"{row['model_name']} v{row['version']}", f"{row['model_type']}"), unsafe_allow_html=True)
with c2:
    st.markdown(kpi_card("Coverage", coverage, "Components with recent predictions"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi_card("Avg confidence", avg_conf, "Recent predictions"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi_card("Validation", val_chip, "Metrics JSON check"), unsafe_allow_html=True)

# ----------------------------
# SUMMARY + EXPORTS
# ----------------------------
left, right = st.columns([1.45, 1.0], gap="large")

with left:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpiTitle">Summary</div>
          <div style="font-weight:850; margin-top:4px;">{row['model_name']} v{row['version']}</div>
          <div class="kpiSub" style="margin-top:10px;">Algorithm: <b>{row['model_type']}</b></div>
          <div class="kpiSub" style="margin-top:6px;">Created: <b>{fmt_dt(row.get('created_at'))}</b></div>
          <div class="kpiSub" style="margin-top:6px;">Last run: <b>{last_run}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="small")
    with a:
        st.download_button(
            "Download metrics (JSON)",
            data=json.dumps(metrics, indent=2) if metrics_ok else "{}",
            file_name=f"model_{selected_model_id}_metrics.json",
            mime="application/json",
        )
    with b:
        st.download_button(
            "Download recent predictions (CSV)",
            data=pred.to_csv(index=False).encode("utf-8") if not pred.empty else b"prediction_time,component_id\n",
            file_name=f"model_{selected_model_id}_recent_predictions.csv",
            mime="text/csv",
        )

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>Performance</div>", unsafe_allow_html=True)

    if not metrics_ok:
        st.markdown("<div class='kpiSub'>Metrics JSON missing required fields.</div>", unsafe_allow_html=True)
        st.markdown("<div class='kpiSub'>Required: precision, recall, accuracy, f1_score.</div>", unsafe_allow_html=True)
    else:
        p = pct_or_dash(metrics.get("precision"))
        r = pct_or_dash(metrics.get("recall"))
        a = pct_or_dash(metrics.get("accuracy"))
        f = pct_or_dash(metrics.get("f1_score"))

        cA, cB = st.columns(2, gap="small")
        with cA:
            st.markdown(kpi_card("Precision", p), unsafe_allow_html=True)
            st.markdown(kpi_card("Accuracy", a), unsafe_allow_html=True)
        with cB:
            st.markdown(kpi_card("Recall", r), unsafe_allow_html=True)
            st.markdown(kpi_card("F1 score", f), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# CHARTS (production signal)
# ----------------------------
st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
cL, cR = st.columns([1.55, 1.0], gap="large")

axis_color, grid_op = altair_axis_colors(dark_mode)

with cL:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>Confidence over time</div>", unsafe_allow_html=True)

    if pred.empty:
        st.markdown("<div class='kpiSub'>No predictions found for this model.</div>", unsafe_allow_html=True)
    else:
        df = pred.copy()
        df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
        df = df.dropna(subset=["prediction_time"])
        # Prefer remaining_life series for clean signal
        if "prediction_type" in df.columns:
            rl = df[df["prediction_type"].astype(str).str.lower() == "remaining_life"]
            if not rl.empty:
                df = rl

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X("prediction_time:T", title="Time",
                        axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
                y=alt.Y("confidence:Q", title="Confidence",
                        axis=alt.Axis(labelColor=axis_color, titleColor=axis_color),
                        scale=alt.Scale(domain=[0, 1])),
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

    st.markdown("</div>", unsafe_allow_html=True)

with cR:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpiTitle'>Prediction mix</div>", unsafe_allow_html=True)

    if pred.empty or "prediction_type" not in pred.columns:
        st.markdown("<div class='kpiSub'>Not available.</div>", unsafe_allow_html=True)
    else:
        mix = pred["prediction_type"].astype(str).str.lower().value_counts().reset_index()
        mix.columns = ["type", "count"]

        bar = (
            alt.Chart(mix)
            .mark_bar()
            .encode(
                x=alt.X("type:N", title="Type",
                        axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
                y=alt.Y("count:Q", title="Count",
                        axis=alt.Axis(labelColor=axis_color, titleColor=axis_color)),
                tooltip=[alt.Tooltip("type:N", title="Type"), alt.Tooltip("count:Q", title="Count")],
            )
            .properties(height=280)
            .configure_view(strokeOpacity=0)
            .configure_axis(gridOpacity=grid_op)
        )
        st.altair_chart(bar, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# AUDIT TABLE
# ----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='kpiTitle'>Recent predictions (audit)</div>", unsafe_allow_html=True)

if pred.empty:
    st.markdown("<div class='kpiSub'>No recent predictions available for this model.</div>", unsafe_allow_html=True)
else:
    comp_map = get_components_map()
    show = pred.copy()

    if not comp_map.empty:
        show = show.merge(comp_map, on="component_id", how="left")
        show["component"] = show.apply(
            lambda r: f"{r.get('tail_number','—')} • {r.get('type','—')} • {r.get('name','—')} #{int(r['component_id'])}",
            axis=1,
        )
    else:
        show["component"] = show["component_id"].astype(str)

    keep = ["prediction_time", "component", "prediction_type", "predicted_value", "confidence", "time_horizon"]
    keep = [c for c in keep if c in show.columns]
    show = show[keep].head(80)

    if "confidence" in show.columns:
        show["confidence"] = pd.to_numeric(show["confidence"], errors="coerce").apply(lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—")
    if "predicted_value" in show.columns:
        show["predicted_value"] = pd.to_numeric(show["predicted_value"], errors="coerce").apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    st.dataframe(show, use_container_width=True, hide_index=True)

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# JSON VALIDATOR (kept, but clean)
# ----------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='kpiTitle'>Validate metrics JSON</div>", unsafe_allow_html=True)
st.markdown("<div class='kpiSub'>Required: precision, recall, accuracy, f1_score.</div>", unsafe_allow_html=True)

user_json = st.text_area(" ", height=160, label_visibility="collapsed")

col1, col2 = st.columns([1, 6])
with col1:
    validate_btn = st.button("Validate")

if validate_btn:
    if validate_metrics(user_json):
        st.success("Valid metrics JSON.")
    else:
        st.error("Missing required fields: precision, recall, accuracy, f1_score.")

st.markdown("</div>", unsafe_allow_html=True)
