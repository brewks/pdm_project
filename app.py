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

          /* -----------------------------
             SIDEBAR: force readable text in BOTH modes
             (fixes: nav items + widget labels invisible in light mode)
          ------------------------------*/
          section[data-testid="stSidebar"] * {{
            color: var(--text) !important;
          }}

          section[data-testid="stSidebar"] [data-testid="stSidebarNav"] * {{
            color: var(--text) !important;
          }}

          section[data-testid="stSidebar"] a,
          section[data-testid="stSidebar"] a * {{
            color: var(--text) !important;
            text-decoration: none;
          }}

          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] label * {{
            color: var(--text) !important;
          }}

          section[data-testid="stSidebar"] [data-baseweb="select"] * {{
            color: var(--text) !important;
          }}
          section[data-testid="stSidebar"] [data-baseweb="select"] input {{
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
          }}

          section[data-testid="stSidebar"] [role="radiogroup"] *,
          section[data-testid="stSidebar"] [data-testid="stCheckbox"] * {{
            color: var(--text) !important;
          }}

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
