import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SeismoSense · Damage Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark background */
.stApp { background: #0d0f14; color: #e8eaf0; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13161e !important;
    border-right: 1px solid #1f2330;
}
[data-testid="stSidebar"] * { color: #c8cad6 !important; }

/* Hide default header */
header[data-testid="stHeader"] { background: transparent; }

/* Hero title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e85d4a 0%, #f5a623 50%, #e85d4a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #6b7280;
    margin-top: 4px;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #e85d4a44; }
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e85d4a;
}
.metric-lbl { font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8eaf0;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-left: 3px solid #e85d4a;
    padding-left: 10px;
    margin: 24px 0 16px 0;
}

/* Input labels */
label { color: #9ca3af !important; font-size: 0.82rem !important; }

/* Selectbox / slider */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input {
    background: #1a1d26 !important;
    border: 1px solid #2a2d3a !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
}

/* Slider track */
[data-testid="stSlider"] [role="slider"] { background: #e85d4a !important; }

/* Checkbox */
[data-testid="stCheckbox"] label { color: #c8cad6 !important; font-size: 0.85rem !important; }

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e85d4a, #f5a623) !important;
    border: none !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 40px !important;
    border-radius: 10px !important;
    letter-spacing: 1px !important;
    transition: opacity 0.2s !important;
    width: 100%;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }

/* Result box */
.result-box {
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 16px;
}
.result-grade {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}
.result-label { font-size: 1rem; margin-top: 8px; opacity: 0.8; }

/* Tab styling */
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e85d4a !important;
    border-bottom-color: #e85d4a !important;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #13161e !important;
    border: 1px dashed #2a2d3a !important;
    border-radius: 12px !important;
}

/* Divider */
hr { border-color: #1f2330 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2a2d3a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("earthquake_model.pkl")

try:
    bundle  = load_model()
    model   = bundle["model"]
    features = bundle["features"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── Constants ─────────────────────────────────────────────────────────────────
CAT_OPTIONS = {
    "land_surface_condition": ["n", "o", "t"],
    "foundation_type":        ["h", "i", "r", "u", "w"],
    "roof_type":              ["n", "q", "x"],
    "ground_floor_type":      ["f", "m", "v", "x", "z"],
    "other_floor_type":       ["j", "q", "s", "x"],
    "position":               ["j", "o", "s", "t"],
    "plan_configuration":     ["a", "c", "d", "f", "m", "n", "o", "q", "s", "u"],
    "legal_ownership_status": ["a", "r", "v", "w"],
}
GRADE_COLOR  = {1: "#22c55e", 2: "#f59e0b", 3: "#e85d4a"}
GRADE_BG     = {1: "#052e16", 2: "#2d1a00", 3: "#2d0a06"}
GRADE_LABEL  = {1: "LOW DAMAGE", 2: "MEDIUM DAMAGE", 3: "NEAR TOTAL DESTRUCTION"}
GRADE_EMOJI  = {1: "✅", 2: "⚠️", 3: "🔴"}

FEATURE_IMPORTANCE = {
    "geo_level_3_id":                     0.179,
    "roof_type":                          0.104,
    "has_superstructure_mud_mortar_stone":0.087,
    "foundation_type":                    0.081,
    "age_x_floors":                       0.074,
    "geo_level_2_id":                     0.068,
    "height_percentage":                  0.061,
    "age":                                0.055,
    "area_x_height":                      0.048,
    "geo_level_1_id":                     0.043,
    "count_floors_pre_eq":                0.039,
    "has_superstructure_rc_engineered":   0.034,
    "area_percentage":                    0.031,
    "ground_floor_type":                  0.028,
    "land_surface_condition":             0.022,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9ca3af", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;
                    color:#e85d4a;letter-spacing:-0.5px'>🌍 SeismoSense</div>
        <div style='font-size:0.75rem;color:#4b5563;margin-top:2px'>
            2015 Gorkha Earthquake · Nepal
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio(
        "Navigation",
        ["🔍 Predict", "📊 Analytics", "📂 Batch", "ℹ️ Model Info"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem;color:#374151;line-height:1.8'>
        <b style='color:#4b5563'>Damage Grades</b><br>
        🟢 Grade 1 · Low<br>
        🟡 Grade 2 · Medium<br>
        🔴 Grade 3 · Near total
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict":

    st.markdown('<div class="hero-title">Building Damage Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter building characteristics to estimate earthquake damage grade</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("⚠️ `earthquake_model.pkl` not found in app directory.")
        st.stop()

    sup_cols = [c for c in features if c.startswith("has_superstructure")]
    use_cols = [c for c in features if c.startswith("has_secondary_use")]

    # ── Form ──────────────────────────────────────────────────────────────────
    col_left, col_mid, col_right = st.columns([1, 1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-head">Geographic</div>', unsafe_allow_html=True)
        geo1 = st.number_input("Geo Level 1 (region mean)", 0.0, 3.0, 2.0, 0.01)
        geo2 = st.number_input("Geo Level 2 (district mean)", 0.0, 3.0, 2.0, 0.01)
        geo3 = st.number_input("Geo Level 3 (ward mean)", 0.0, 3.0, 2.0, 0.01)

        st.markdown('<div class="section-head">Building Basics</div>', unsafe_allow_html=True)
        floors    = st.slider("Floors before earthquake", 1, 10, 2)
        age       = st.slider("Building age (years)", 0, 200, 20)
        area_pct  = st.slider("Area percentage", 1, 100, 10)
        height_pct= st.slider("Height percentage", 1, 100, 5)
        families  = st.slider("Number of families", 0, 20, 1)

    with col_mid:
        st.markdown('<div class="section-head">Structural</div>', unsafe_allow_html=True)
        land  = st.selectbox("Land surface condition",  CAT_OPTIONS["land_surface_condition"])
        found = st.selectbox("Foundation type",         CAT_OPTIONS["foundation_type"])
        roof  = st.selectbox("Roof type",               CAT_OPTIONS["roof_type"])
        gfloor= st.selectbox("Ground floor type",       CAT_OPTIONS["ground_floor_type"])
        ofloor= st.selectbox("Other floor type",        CAT_OPTIONS["other_floor_type"])
        pos   = st.selectbox("Position",                CAT_OPTIONS["position"])
        plan  = st.selectbox("Plan configuration",      CAT_OPTIONS["plan_configuration"])
        legal = st.selectbox("Legal ownership status",  CAT_OPTIONS["legal_ownership_status"])

    with col_right:
        st.markdown('<div class="section-head">Superstructure Materials</div>', unsafe_allow_html=True)
        sup_vals = {}
        for col in sup_cols:
            label = col.replace("has_superstructure_", "").replace("_", " ").title()
            sup_vals[col] = int(st.checkbox(label, key=f"sup_{col}"))

        st.markdown('<div class="section-head">Secondary Use</div>', unsafe_allow_html=True)
        use_vals = {}
        for col in use_cols:
            label = col.replace("has_secondary_use_", "").replace("_", " ").title()
            use_vals[col] = int(st.checkbox(label, key=f"use_{col}"))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ PREDICT DAMAGE GRADE", type="primary")

    if predict_btn:
        vals = {
            "geo_level_1_id":         geo1,
            "geo_level_2_id":         geo2,
            "geo_level_3_id":         geo3,
            "count_floors_pre_eq":    floors,
            "age":                    age,
            "area_percentage":        area_pct,
            "height_percentage":      height_pct,
            "count_families":         families,
            "land_surface_condition": land,
            "foundation_type":        found,
            "roof_type":              roof,
            "ground_floor_type":      gfloor,
            "other_floor_type":       ofloor,
            "position":               pos,
            "plan_configuration":     plan,
            "legal_ownership_status": legal,
            "age_x_floors":           age * floors,
            "area_x_height":          area_pct * height_pct,
            **sup_vals, **use_vals,
        }
        for f in features:
            if f not in vals:
                vals[f] = 0

        input_df = pd.DataFrame([vals])[features]
        pred     = int(model.predict(input_df)[0]) + 1
        proba    = model.predict_proba(input_df)[0]

        st.divider()
        r1, r2 = st.columns([1, 2], gap="large")

        with r1:
            color = GRADE_COLOR[pred]
            bg    = GRADE_BG[pred]
            st.markdown(f"""
            <div class="result-box" style="background:{bg};border:1.5px solid {color}44">
                <div style="font-size:3.5rem">{GRADE_EMOJI[pred]}</div>
                <div class="result-grade" style="color:{color}">Grade {pred}</div>
                <div class="result-label" style="color:{color}">{GRADE_LABEL[pred]}</div>
                <div style="margin-top:16px;font-size:0.78rem;color:#6b7280">
                    Confidence: <b style="color:{color}">{proba[pred-1]*100:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            fig = go.Figure()
            grades = ["Grade 1 · Low", "Grade 2 · Medium", "Grade 3 · High"]
            colors_bar = ["#22c55e", "#f59e0b", "#e85d4a"]
            for i, (g, c, p) in enumerate(zip(grades, colors_bar, proba)):
                fig.add_trace(go.Bar(
                    x=[p * 100], y=[g], orientation="h",
                    marker=dict(color=c, opacity=1.0 if i == pred-1 else 0.35),
                    text=f"{p*100:.1f}%", textposition="outside",
                    textfont=dict(color=c, size=13, family="Syne"),
                    name=g, showlegend=False,
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text="Probability Distribution", font=dict(family="Syne", size=14, color="#e8eaf0")),
                xaxis=dict(range=[0, 110], showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(size=12)),
                height=220, barmode="overlay",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Risk factors
            risk_notes = []
            if age > 50:   risk_notes.append("🔴 Old building (>50 yrs)")
            if floors > 4: risk_notes.append("🔴 High-rise (>4 floors)")
            if sup_vals.get("has_superstructure_mud_mortar_stone", 0): risk_notes.append("🔴 Mud mortar stone material")
            if sup_vals.get("has_superstructure_rc_engineered", 0):    risk_notes.append("🟢 RC engineered (protective)")
            if roof == "n": risk_notes.append("⚠️ Roof type N (higher risk)")
            if found in ["r", "h"]: risk_notes.append("⚠️ Foundation may be vulnerable")

            if risk_notes:
                st.markdown('<div class="section-head" style="margin-top:8px">Risk Factors</div>', unsafe_allow_html=True)
                for note in risk_notes:
                    st.markdown(f"<div style='font-size:0.85rem;padding:3px 0;color:#c8cad6'>{note}</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown('<div class="hero-title">Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Performance metrics · Feature importance · Dataset insights</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("75.29%", "Accuracy"),
        ("0.7046",  "F1 Macro"),
        ("±0.003",  "CV Std Dev"),
        ("260K",    "Training Rows"),
        ("40",      "Features"),
    ]
    for col, (val, lbl) in zip([k1,k2,k3,k4,k5], kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  Feature Importance  ", "  Model Comparison  ", "  Dataset Stats  "])

    with tab1:
        fi_df = pd.DataFrame(list(FEATURE_IMPORTANCE.items()), columns=["Feature", "Importance"]).sort_values("Importance")
        fi_df["Feature"] = fi_df["Feature"].str.replace("has_superstructure_", "sup: ").str.replace("_", " ")

        fig = go.Figure(go.Bar(
            x=fi_df["Importance"],
            y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0, "#1f2330"], [0.5, "#f59e0b"], [1, "#e85d4a"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in fi_df["Importance"]],
            textposition="outside",
            textfont=dict(color="#9ca3af", size=11),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=520,
            title=dict(text="Top 15 Feature Importances — XGBoost Tuned", font=dict(family="Syne", size=15, color="#e8eaf0")),
            xaxis=dict(showgrid=True, gridcolor="#1f2330", zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        models = ["Logistic Reg", "Decision Tree", "Random Forest", "CatBoost", "XGBoost Base", "XGBoost Tuned"]
        acc    = [0.6891, 0.6476, 0.7311, 0.7497, 0.7501, 0.7529]
        f1     = [0.6612, 0.5975, 0.6761, 0.7040, 0.6997, 0.7046]

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "F1 Macro"),
                            horizontal_spacing=0.12)
        bar_colors = ["#2a2d3a","#2a2d3a","#2a2d3a","#2a2d3a","#2a2d3a","#e85d4a"]

        fig.add_trace(go.Bar(x=models, y=acc, marker_color=bar_colors,
                             text=[f"{v:.4f}" for v in acc], textposition="outside",
                             textfont=dict(size=10, color="#9ca3af"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=models, y=f1, marker_color=bar_colors,
                             text=[f"{v:.4f}" for v in f1], textposition="outside",
                             textfont=dict(size=10, color="#9ca3af"), showlegend=False), row=1, col=2)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=400,
            yaxis=dict(range=[0.4, 0.82], showgrid=True, gridcolor="#1f2330"),
            yaxis2=dict(range=[0.4, 0.78], showgrid=True, gridcolor="#1f2330"),
        )
        fig.update_annotations(font=dict(family="Syne", size=13, color="#e8eaf0"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='background:#13161e;border:1px solid #1f2330;border-radius:10px;padding:16px 20px;font-size:0.85rem;color:#9ca3af;line-height:2'>
        <b style='color:#e85d4a;font-family:Syne'>✅ Winner: XGBoost Tuned</b><br>
        Best params: <code>learning_rate=0.05 · max_depth=10 · n_estimators=300 · subsample=0.8</code><br>
        5-fold CV Mean F1: <b style='color:#f5a623'>0.7038 ± 0.003</b> · Overfitting gap: 0.054 (acceptable)
        </div>""", unsafe_allow_html=True)

    with tab3:
        d1, d2 = st.columns(2)
        with d1:
            # Class distribution
            labels = ["Grade 2 · Medium", "Grade 3 · High", "Grade 1 · Low"]
            values = [56.9, 33.5, 9.6]
            colors_pie = ["#f59e0b", "#e85d4a", "#22c55e"]
            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.55,
                marker=dict(colors=colors_pie, line=dict(color="#0d0f14", width=2)),
                textinfo="label+percent",
                textfont=dict(size=11, family="DM Sans"),
            ))
            fig.add_annotation(text="<b>260K</b><br>buildings", x=0.5, y=0.5,
                               font=dict(size=13, family="Syne", color="#e8eaf0"),
                               showarrow=False)
            fig.update_layout(**PLOTLY_LAYOUT, height=340,
                              title=dict(text="Target Class Distribution", font=dict(family="Syne", size=14, color="#e8eaf0")),
                              legend=dict(font=dict(size=11)))
            st.plotly_chart(fig, use_container_width=True)

        with d2:
            # Grade-level metrics
            grades_m = ["Grade 1", "Grade 2", "Grade 3"]
            precision = [0.58, 0.77, 0.72]
            recall    = [0.55, 0.82, 0.64]
            f1_per    = [0.56, 0.79, 0.68]
            x = np.arange(len(grades_m))
            fig = go.Figure()
            for name, vals, col in zip(["Precision","Recall","F1"], [precision,recall,f1_per], ["#e85d4a","#f59e0b","#22c55e"]):
                fig.add_trace(go.Bar(name=name, x=grades_m, y=vals,
                                     marker_color=col, opacity=0.85,
                                     text=[f"{v:.2f}" for v in vals],
                                     textposition="outside",
                                     textfont=dict(size=10)))
            fig.update_layout(
                **PLOTLY_LAYOUT, barmode="group", height=340,
                title=dict(text="Per-Class Metrics (XGBoost Tuned)", font=dict(family="Syne", size=14, color="#e8eaf0")),
                yaxis=dict(range=[0, 1.1], showgrid=True, gridcolor="#1f2330"),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Batch":

    st.markdown('<div class="hero-title">Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a CSV to predict damage grades for multiple buildings at once</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.error("⚠️ `earthquake_model.pkl` not found.")
        st.stop()

    st.markdown("""
    <div style='background:#13161e;border:1px solid #1f2330;border-radius:10px;
                padding:14px 20px;font-size:0.83rem;color:#6b7280;margin-bottom:20px'>
    📌 CSV must contain the same columns as training data.
    Geo columns (<code>geo_level_1/2/3_id</code>) should be <b>target-encoded float values</b> (mean damage grade per region).
    Missing columns will be filled with 0.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop your CSV here", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f'<div class="section-head">Preview — {len(df):,} rows</div>', unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True)

        if "age_x_floors" not in df.columns:
            df["age_x_floors"] = df["age"] * df["count_floors_pre_eq"]
        if "area_x_height" not in df.columns:
            df["area_x_height"] = df["area_percentage"] * df["height_percentage"]
        for f in features:
            if f not in df.columns:
                df[f] = 0

        preds = model.predict(df[features]) + 1
        df["predicted_damage_grade"] = preds
        df["damage_label"] = df["predicted_damage_grade"].map({1:"Low", 2:"Medium", 3:"Almost Complete"})

        st.divider()
        b1, b2, b3 = st.columns(3)
        counts = df["predicted_damage_grade"].value_counts()
        for col, grade, label in zip([b1,b2,b3], [1,2,3], ["Low","Medium","High"]):
            n = counts.get(grade, 0)
            pct = n / len(df) * 100
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:{GRADE_COLOR[grade]}44">
                    <div class="metric-val" style="color:{GRADE_COLOR[grade]}">{n:,}</div>
                    <div class="metric-lbl">Grade {grade} · {label} ({pct:.1f}%)</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=[f"Grade {g}" for g in [1,2,3]],
            values=[counts.get(g,0) for g in [1,2,3]],
            hole=0.5, marker=dict(colors=["#22c55e","#f59e0b","#e85d4a"],
                                  line=dict(color="#0d0f14", width=2)),
            textinfo="label+percent",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=300,
                          title=dict(text="Predicted Damage Distribution", font=dict(family="Syne", size=14, color="#e8eaf0")))
        st.plotly_chart(fig, use_container_width=True)

        csv_out = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions CSV", csv_out, "predictions.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ Model Info":

    st.markdown('<div class="hero-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Architecture · Training details · Key findings</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    i1, i2 = st.columns(2, gap="large")

    with i1:
        st.markdown('<div class="section-head">Model Summary</div>', unsafe_allow_html=True)
        rows = [
            ("Dataset",    "2015 Gorkha Earthquake, Nepal"),
            ("Buildings",  "260,601 (248,282 after dedup)"),
            ("Task",       "Multiclass ordinal classification"),
            ("Algorithm",  "XGBoost (tuned)"),
            ("Accuracy",   "0.7529"),
            ("F1 Macro",   "0.7046"),
            ("CV Mean F1", "0.7038 ± 0.003"),
            ("Features",   "40 (incl. 2 engineered)"),
        ]
        for k, v in rows:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:8px 0;
                        border-bottom:1px solid #1f2330;font-size:0.85rem'>
                <span style='color:#6b7280'>{k}</span>
                <span style='color:#e8eaf0;font-weight:500'>{v}</span>
            </div>""", unsafe_allow_html=True)

        if MODEL_LOADED:
            st.markdown('<div class="section-head" style="margin-top:24px">Best Hyperparameters</div>', unsafe_allow_html=True)
            for k, v in bundle.get("best_params", {}).items():
                k_clean = k.replace("model__", "")
                st.markdown(f"""
                <div style='display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid #1f2330;font-size:0.84rem'>
                    <span style='color:#6b7280;font-family:monospace'>{k_clean}</span>
                    <span style='color:#f5a623;font-weight:600'>{v}</span>
                </div>""", unsafe_allow_html=True)

    with i2:
        st.markdown('<div class="section-head">Key Findings</div>', unsafe_allow_html=True)
        findings = [
            ("🗺️", "Geographic location (geo_level_3_id) is the strongest damage predictor (0.179 importance)"),
            ("🏠", "Roof type is 2nd most important — type N roofs show highest damage correlation"),
            ("🧱", "Mud mortar stone superstructures strongly linked to Grade 2/3 damage"),
            ("🔬", "RC engineered construction significantly reduces damage risk"),
            ("📅", "Older + taller buildings most vulnerable (age × floors interaction feature)"),
            ("🎯", "Target encoding on geo columns boosted accuracy from ~0.65 → ~0.75"),
        ]
        for icon, text in findings:
            st.markdown(f"""
            <div style='display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #1f2330'>
                <span style='font-size:1.2rem'>{icon}</span>
                <span style='font-size:0.84rem;color:#9ca3af;line-height:1.5'>{text}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-head" style="margin-top:24px">Pipeline Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#13161e;border:1px solid #1f2330;border-radius:10px;
                    padding:16px;font-size:0.8rem;font-family:monospace;color:#9ca3af;line-height:2'>
        Pipeline(<br>
        &nbsp;&nbsp;ColumnTransformer(<br>
        &nbsp;&nbsp;&nbsp;&nbsp;num: Imputer(median) → StandardScaler<br>
        &nbsp;&nbsp;&nbsp;&nbsp;cat: Imputer(mode) → OrdinalEncoder<br>
        &nbsp;&nbsp;),<br>
        &nbsp;&nbsp;<span style='color:#e85d4a'>XGBClassifier(n_est=300, lr=0.05, depth=10)</span><br>
        )
        </div>""", unsafe_allow_html=True)
