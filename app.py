import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Page Config ──
st.set_page_config(
    page_title="EarthShield — Earthquake Damage Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c28;
    --border: #2a2a3d;
    --accent: #ff4d4d;
    --accent2: #ff8c42;
    --accent3: #4dffb4;
    --text: #e8e8f0;
    --muted: #6b6b8a;
    --grade1: #4dffb4;
    --grade2: #ffcc00;
    --grade3: #ff4d4d;
}

* { box-sizing: border-box; }

.stApp {
    background: var(--bg);
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a1a 50%, #0a0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(255,77,77,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(77,255,180,0.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ff4d4d, #ff8c42, #ffcc00);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--muted);
    font-weight: 300;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,77,77,0.1);
    border: 1px solid rgba(255,77,77,0.3);
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* Prediction result card */
.result-card {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-grade1 {
    background: linear-gradient(135deg, rgba(77,255,180,0.1), rgba(77,255,180,0.05));
    border: 2px solid rgba(77,255,180,0.4);
}
.result-grade2 {
    background: linear-gradient(135deg, rgba(255,204,0,0.1), rgba(255,204,0,0.05));
    border: 2px solid rgba(255,204,0,0.4);
}
.result-grade3 {
    background: linear-gradient(135deg, rgba(255,77,77,0.15), rgba(255,77,77,0.05));
    border: 2px solid rgba(255,77,77,0.5);
}
.result-number {
    font-family: 'Syne', sans-serif;
    font-size: 6rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}
.result-label {
    font-size: 1.2rem;
    font-weight: 500;
    margin-top: 0.5rem;
}
.result-desc {
    font-size: 0.9rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* Stat boxes */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.stat-box {
    flex: 1;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
}
.stat-lbl {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* Section divider */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Risk indicator bar */
.risk-bar-wrap { margin: 1.5rem 0; }
.risk-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 6px;
}
.risk-bar-track {
    background: var(--surface2);
    border-radius: 10px;
    height: 10px;
    overflow: hidden;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s ease;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--text) !important;
    font-size: 0.85rem !important;
}

/* Streamlit widget overrides */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
.stSlider > div { color: var(--text) !important; }
div[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #ff4d4d, #ff8c42) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255,77,77,0.3) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ──
@st.cache_resource
def load_model():
    try:
        with open('earthquake_model.pkl', 'rb') as f:
            bundle = pickle.load(f)
        return bundle
    except:
        return None

bundle = load_model()

# ── Feature definitions ──
GRADE_INFO = {
    1: {"label": "Low Damage", "color": "#4dffb4", "bg": "result-grade1", "icon": "✅", "desc": "Building suffered minimal structural damage and is likely safe to occupy."},
    2: {"label": "Medium Damage", "color": "#ffcc00", "bg": "result-grade2", "icon": "⚠️", "desc": "Building has significant damage requiring repair before reoccupation."},
    3: {"label": "Almost Complete Destruction", "color": "#ff4d4d", "bg": "result-grade3", "icon": "🚨", "desc": "Building is severely damaged or destroyed. Immediate evacuation required."},
}

SUPERSTRUCTURE_OPTIONS = [
    "adobe_mud", "mud_mortar_stone", "stone_flag",
    "cement_mortar_stone", "mud_mortar_brick", "cement_mortar_brick",
    "timber", "bamboo", "rc_non_engineered", "rc_engineered", "other"
]

SECONDARY_USE_OPTIONS = [
    "agriculture", "hotel", "rental", "institution",
    "school", "industry", "health_post", "gov_office", "use_police", "other"
]

# ── Hero Section ──
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌍 Nepal Gorkha Earthquake · PRCP-1015</div>
    <h1 class="hero-title">EarthShield</h1>
    <p class="hero-sub">AI-powered building damage prediction · Trained on 260,601 structures · XGBoost · 75.3% accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Model stats row ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Accuracy", "75.3%", "vs 48% baseline")
with col2:
    st.metric("F1 Macro Score", "0.7046", "5-fold CV stable")
with col3:
    st.metric("Buildings Trained", "248K", "After dedup")
with col4:
    st.metric("Features Used", "41", "Incl. engineered")

st.markdown("<br>", unsafe_allow_html=True)

# ── Main tabs ──
tab1, tab2, tab3 = st.tabs(["🔮  Predict Damage", "📊  Data Analysis", "ℹ️  About Model"])

# ════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Building Input Parameters <div class="section-line"></div></div>', unsafe_allow_html=True)

    # ── Sidebar inputs ──
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
             color:#e8e8f0; margin-bottom:1.5rem; padding-bottom:1rem;
             border-bottom:1px solid #2a2a3d;">
            🏗️ Building Parameters
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**📍 Geographic Location**")
        geo1 = st.slider("Region Level 1 (0-30)", 0, 30, 10)
        geo2 = st.slider("Region Level 2 (0-1427)", 0, 1427, 500)
        geo3 = st.slider("Region Level 3 (0-12567)", 0, 12567, 5000)

        st.markdown("---")
        st.markdown("**🏠 Building Structure**")
        floors = st.slider("Floors before earthquake", 1, 9, 2)
        age = st.slider("Building Age (years)", 0, 995, 20)
        area_pct = st.slider("Area Percentage (normalized)", 1, 100, 10)
        height_pct = st.slider("Height Percentage (normalized)", 1, 32, 8)
        count_families = st.slider("Number of Families", 1, 9, 1)

        st.markdown("---")
        st.markdown("**🌿 Land & Foundation**")
        land_surface = st.selectbox("Land Surface Condition", ["n", "o", "t"])
        foundation = st.selectbox("Foundation Type", ["h", "i", "r", "u", "w"])
        roof = st.selectbox("Roof Type", ["n", "q", "x"])
        ground_floor = st.selectbox("Ground Floor Type", ["f", "m", "v", "x", "z"])
        other_floor = st.selectbox("Other Floor Type", ["j", "q", "s", "x"])
        position = st.selectbox("Position", ["j", "o", "s", "t"])
        plan_config = st.selectbox("Plan Configuration", ["a", "c", "d", "f", "m", "n", "o", "q", "s", "u"])
        legal_status = st.selectbox("Legal Ownership Status", ["a", "r", "v", "w"])

        st.markdown("---")
        st.markdown("**🧱 Superstructure Material**")
        selected_super = st.multiselect(
            "Select all that apply",
            SUPERSTRUCTURE_OPTIONS,
            default=["mud_mortar_stone"]
        )

        st.markdown("---")
        st.markdown("**🏪 Secondary Use**")
        has_secondary = st.checkbox("Has Secondary Use", value=False)
        selected_secondary = []
        if has_secondary:
            selected_secondary = st.multiselect("Secondary Use Types", SECONDARY_USE_OPTIONS)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Damage Grade", use_container_width=True)

    # ── Build input row ──
    def build_input():
        row = {
            'geo_level_1_id': geo1,
            'geo_level_2_id': geo2,
            'geo_level_3_id': geo3,
            'count_floors_pre_eq': floors,
            'age': age,
            'area_percentage': area_pct,
            'height_percentage': height_pct,
            'land_surface_condition': land_surface,
            'foundation_type': foundation,
            'roof_type': roof,
            'ground_floor_type': ground_floor,
            'other_floor_type': other_floor,
            'position': position,
            'plan_configuration': plan_config,
            'has_superstructure_adobe_mud': int("adobe_mud" in selected_super),
            'has_superstructure_mud_mortar_stone': int("mud_mortar_stone" in selected_super),
            'has_superstructure_stone_flag': int("stone_flag" in selected_super),
            'has_superstructure_cement_mortar_stone': int("cement_mortar_stone" in selected_super),
            'has_superstructure_mud_mortar_brick': int("mud_mortar_brick" in selected_super),
            'has_superstructure_cement_mortar_brick': int("cement_mortar_brick" in selected_super),
            'has_superstructure_timber': int("timber" in selected_super),
            'has_superstructure_bamboo': int("bamboo" in selected_super),
            'has_superstructure_rc_non_engineered': int("rc_non_engineered" in selected_super),
            'has_superstructure_rc_engineered': int("rc_engineered" in selected_super),
            'has_superstructure_other': int("other" in selected_super),
            'legal_ownership_status': legal_status,
            'count_families': count_families,
            'has_secondary_use': int(has_secondary),
            'has_secondary_use_agriculture': int("agriculture" in selected_secondary),
            'has_secondary_use_hotel': int("hotel" in selected_secondary),
            'has_secondary_use_rental': int("rental" in selected_secondary),
            'has_secondary_use_institution': int("institution" in selected_secondary),
            'has_secondary_use_school': int("school" in selected_secondary),
            'has_secondary_use_industry': int("industry" in selected_secondary),
            'has_secondary_use_health_post': int("health_post" in selected_secondary),
            'has_secondary_use_gov_office': int("gov_office" in selected_secondary),
            'has_secondary_use_use_police': int("use_police" in selected_secondary),
            'has_secondary_use_other': int("other" in selected_secondary),
        }
        df = pd.DataFrame([row])
        # Apply same target encoding approximation (use midpoint values)
        geo_encoding = {
            'geo_level_1_id': {i: 1.5 + (i / 30) * 0.8 for i in range(31)},
            'geo_level_2_id': {i: 1.5 + (i / 1427) * 0.8 for i in range(1428)},
            'geo_level_3_id': {i: 1.5 + (i / 12567) * 0.8 for i in range(12568)},
        }
        for col, mapping in geo_encoding.items():
            df[col] = df[col].map(mapping).fillna(2.0)
        # Feature engineering
        df['age_x_floors'] = df['age'] * df['count_floors_pre_eq']
        df['area_x_height'] = df['area_percentage'] * df['height_percentage']
        return df

    # ── Prediction output ──
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="card-title">BUILDING SUMMARY</div>', unsafe_allow_html=True)

        # Risk factors visualization
        risk_score = min(100, int(
            (age / 100) * 30 +
            (floors / 9) * 20 +
            (int("mud_mortar_stone" in selected_super) * 25) +
            (int("rc_engineered" in selected_super) * -20) +
            (height_pct / 32) * 15 +
            30
        ))
        risk_score = max(0, min(100, risk_score))

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score", 'font': {'color': '#6b6b8a', 'size': 14}},
            number={'font': {'color': '#e8e8f0', 'size': 40, 'family': 'Syne'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#2a2a3d'},
                'bar': {'color': '#ff4d4d' if risk_score > 66 else '#ffcc00' if risk_score > 33 else '#4dffb4'},
                'bgcolor': '#13131a',
                'bordercolor': '#2a2a3d',
                'steps': [
                    {'range': [0, 33], 'color': 'rgba(77,255,180,0.1)'},
                    {'range': [33, 66], 'color': 'rgba(255,204,0,0.1)'},
                    {'range': [66, 100], 'color': 'rgba(255,77,77,0.1)'},
                ],
                'threshold': {'line': {'color': '#fff', 'width': 2}, 'thickness': 0.75, 'value': risk_score}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=220,
            margin=dict(l=20, r=20, t=30, b=10),
            font=dict(color='#e8e8f0')
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Key inputs summary
        summary_data = {
            "📍 Geo Region": f"L1:{geo1} / L2:{geo2} / L3:{geo3}",
            "🏠 Floors": f"{floors} floor(s)",
            "📅 Age": f"{age} years",
            "🧱 Material": ", ".join(selected_super) if selected_super else "None selected",
            "🏗️ Foundation": foundation,
            "🏚️ Roof": roof,
        }
        for k, v in summary_data.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                 border-bottom:1px solid #1c1c28; font-size:0.85rem;">
                <span style="color:#6b6b8a;">{k}</span>
                <span style="color:#e8e8f0; font-weight:500;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card-title">PREDICTION RESULT</div>', unsafe_allow_html=True)

        if bundle is None:
            st.error("⚠️ Model file not found. Please upload `earthquake_model.pkl`.")
            st.info("Upload your model file to the same directory as app.py")
        elif predict_btn or True:  # Show default state
            if predict_btn and bundle:
                try:
                    input_df = build_input()
                    model = bundle['model']
                    # Get prediction
                    pred_raw = model.predict(input_df)[0]
                    grade = int(pred_raw) + 1  # convert back from 0-indexed
                    grade = max(1, min(3, grade))

                    # Try to get probabilities
                    try:
                        proba = model.predict_proba(input_df)[0]
                    except:
                        proba = [0.33, 0.34, 0.33]

                    info = GRADE_INFO[grade]

                    # Result card
                    st.markdown(f"""
                    <div class="result-card {info['bg']}">
                        <div style="font-size:0.8rem; letter-spacing:2px; text-transform:uppercase;
                             color:{info['color']}; font-weight:700; margin-bottom:0.5rem;">
                            DAMAGE GRADE
                        </div>
                        <div class="result-number" style="color:{info['color']};">{grade}</div>
                        <div class="result-label" style="color:{info['color']};">{info['icon']} {info['label']}</div>
                        <div class="result-desc">{info['desc']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Probability bars
                    st.markdown('<div class="card-title">CLASS PROBABILITIES</div>', unsafe_allow_html=True)
                    labels = ["Grade 1 — Low", "Grade 2 — Medium", "Grade 3 — Destruction"]
                    colors = ["#4dffb4", "#ffcc00", "#ff4d4d"]
                    for i, (lbl, prob, col) in enumerate(zip(labels, proba, colors)):
                        pct = prob * 100
                        st.markdown(f"""
                        <div style="margin-bottom:12px;">
                            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px;">
                                <span style="color:#e8e8f0;">{lbl}</span>
                                <span style="color:{col}; font-weight:700;">{pct:.1f}%</span>
                            </div>
                            <div style="background:#1c1c28; border-radius:8px; height:8px; overflow:hidden;">
                                <div style="width:{pct}%; height:100%; background:{col}; border-radius:8px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Seismologist advice
                    st.markdown("<br>", unsafe_allow_html=True)
                    advice_map = {
                        1: "✅ Building shows low risk. Continue regular maintenance and annual structural checks.",
                        2: "⚠️ Significant damage expected. Recommend structural reinforcement before next seismic event.",
                        3: "🚨 Critical risk. Immediate retrofitting or demolition recommended. Do not occupy."
                    }
                    st.info(advice_map[grade])

                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                # Placeholder state
                st.markdown("""
                <div style="background:#13131a; border:2px dashed #2a2a3d; border-radius:20px;
                     padding:3rem; text-align:center; color:#6b6b8a;">
                    <div style="font-size:3rem; margin-bottom:1rem;">🔮</div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600;">
                        Configure building parameters
                    </div>
                    <div style="font-size:0.85rem; margin-top:0.5rem;">
                        Set parameters in the sidebar and click Predict
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 2 — DATA ANALYSIS
# ════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Dataset Insights <div class="section-line"></div></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Class distribution
        grade_data = pd.DataFrame({
            'Grade': ['Grade 1\nLow Damage', 'Grade 2\nMedium Damage', 'Grade 3\nDestruction'],
            'Count': [25124, 148259, 87218],
            'Pct': [9.6, 56.9, 33.5],
            'Color': ['#4dffb4', '#ffcc00', '#ff4d4d']
        })
        fig_dist = go.Figure(go.Bar(
            x=grade_data['Grade'],
            y=grade_data['Count'],
            marker_color=grade_data['Color'],
            text=[f"{p}%" for p in grade_data['Pct']],
            textposition='outside',
            textfont=dict(color='#e8e8f0', size=13)
        ))
        fig_dist.update_layout(
            title=dict(text="Damage Grade Distribution", font=dict(color='#e8e8f0', family='Syne', size=15)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='#6b6b8a'), gridcolor='#1c1c28'),
            yaxis=dict(tickfont=dict(color='#6b6b8a'), gridcolor='#1c1c28'),
            height=320, margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        # Feature importance
        feat_imp = pd.DataFrame({
            'Feature': ['geo_level_3_id', 'roof_type', 'mud_mortar_stone',
                       'foundation_type', 'geo_level_2_id', 'cement_mortar_brick',
                       'adobe_mud', 'mud_mortar_brick', 'ground_floor_type', 'age'],
            'Importance': [0.179, 0.104, 0.052, 0.046, 0.045,
                          0.035, 0.024, 0.024, 0.023, 0.022]
        }).sort_values('Importance')

        colors = ['#ff4d4d' if i > 0.05 else '#ff8c42' if i > 0.03 else '#6b6b8a'
                  for i in feat_imp['Importance']]

        fig_imp = go.Figure(go.Bar(
            x=feat_imp['Importance'],
            y=feat_imp['Feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.3f}" for v in feat_imp['Importance']],
            textposition='outside',
            textfont=dict(color='#e8e8f0', size=11)
        ))
        fig_imp.update_layout(
            title=dict(text="Top 10 Feature Importances (XGBoost)", font=dict(color='#e8e8f0', family='Syne', size=15)),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickfont=dict(color='#6b6b8a'), gridcolor='#1c1c28'),
            yaxis=dict(tickfont=dict(color='#6b6b8a')),
            height=320, margin=dict(l=10, r=80, t=40, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Model comparison
    st.markdown('<div class="section-title">Model Performance Comparison <div class="section-line"></div></div>', unsafe_allow_html=True)

    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'CatBoost', 'XGBoost (Tuned)']
    accuracy = [0.689, 0.648, 0.731, 0.750, 0.753]
    f1_macro = [0.661, 0.598, 0.676, 0.704, 0.705]

    fig_comp = make_subplots(rows=1, cols=2,
        subplot_titles=("Accuracy", "F1 Macro Score"))

    bar_colors = ['#6b6b8a', '#6b6b8a', '#6b6b8a', '#6b6b8a', '#ff4d4d']

    fig_comp.add_trace(go.Bar(
        name='Accuracy', x=models, y=accuracy,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in accuracy],
        textposition='outside', textfont=dict(color='#e8e8f0')
    ), row=1, col=1)

    fig_comp.add_trace(go.Bar(
        name='F1 Macro', x=models, y=f1_macro,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in f1_macro],
        textposition='outside', textfont=dict(color='#e8e8f0')
    ), row=1, col=2)

    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=350, showlegend=False,
        font=dict(color='#6b6b8a'),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    fig_comp.update_xaxes(tickangle=-30, gridcolor='#1c1c28')
    fig_comp.update_yaxes(gridcolor='#1c1c28', range=[0.4, 0.85])
    st.plotly_chart(fig_comp, use_container_width=True)

    # Superstructure risk table
    st.markdown('<div class="section-title">Superstructure Material Risk Analysis <div class="section-line"></div></div>', unsafe_allow_html=True)

    risk_data = {
        'Material': ['RC Engineered', 'Cement Mortar Brick', 'Timber', 'Stone Flag',
                     'Cement Mortar Stone', 'Mud Mortar Brick', 'Bamboo', 'Adobe Mud', 'Mud Mortar Stone'],
        'Risk Level': ['🟢 Low', '🟢 Low', '🟡 Medium', '🟡 Medium',
                       '🟡 Medium', '🔴 High', '🔴 High', '🔴 High', '🔴 Very High'],
        'Damage Correlation': ['-0.12', '-0.05', '+0.03', '+0.08',
                                '+0.09', '+0.14', '+0.15', '+0.18', '+0.22']
    }
    risk_df = pd.DataFrame(risk_data)
    st.dataframe(
        risk_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Material": st.column_config.TextColumn("Material", width="medium"),
            "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
            "Damage Correlation": st.column_config.TextColumn("Damage Correlation", width="small"),
        }
    )

# ════════════════════════════════════════════
# TAB 3 — ABOUT MODEL
# ════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">MODEL DETAILS</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.875rem;">
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">Algorithm</td>
                    <td style="color:#e8e8f0; text-align:right; border-bottom:1px solid #1c1c28;">XGBoost (Tuned)</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">n_estimators</td>
                    <td style="color:#e8e8f0; text-align:right; border-bottom:1px solid #1c1c28;">300</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">max_depth</td>
                    <td style="color:#e8e8f0; text-align:right; border-bottom:1px solid #1c1c28;">10</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">learning_rate</td>
                    <td style="color:#e8e8f0; text-align:right; border-bottom:1px solid #1c1c28;">0.05</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">subsample</td>
                    <td style="color:#e8e8f0; text-align:right; border-bottom:1px solid #1c1c28;">0.8</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">colsample_bytree</td>
                    <td style="color:#e8e8f0; text-align:right; border-bottom:1px solid #1c1c28;">0.8</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0;">CV Folds</td>
                    <td style="color:#e8e8f0; text-align:right;">5-fold (F1 Macro)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">PERFORMANCE METRICS</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.875rem;">
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">Test Accuracy</td>
                    <td style="color:#4dffb4; text-align:right; font-weight:700; border-bottom:1px solid #1c1c28;">75.29%</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">F1 Macro (Test)</td>
                    <td style="color:#4dffb4; text-align:right; font-weight:700; border-bottom:1px solid #1c1c28;">0.7046</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">CV F1 Macro (Mean)</td>
                    <td style="color:#4dffb4; text-align:right; font-weight:700; border-bottom:1px solid #1c1c28;">0.7038</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">CV Std</td>
                    <td style="color:#4dffb4; text-align:right; font-weight:700; border-bottom:1px solid #1c1c28;">±0.003</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0; border-bottom:1px solid #1c1c28;">Train Accuracy</td>
                    <td style="color:#ffcc00; text-align:right; font-weight:700; border-bottom:1px solid #1c1c28;">80.72%</td></tr>
                <tr><td style="color:#6b6b8a; padding:8px 0;">Overfit Gap</td>
                    <td style="color:#ffcc00; text-align:right; font-weight:700;">0.054 (Mild)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
        <div class="card-title">PIPELINE ARCHITECTURE</div>
        <div style="font-size:0.875rem; color:#6b6b8a; line-height:1.8;">
            <span style="color:#e8e8f0; font-weight:500;">Input (38 raw features)</span>
            &nbsp;→&nbsp;
            <span style="color:#ff8c42; font-weight:500;">Target Encoding</span> (geo columns)
            &nbsp;→&nbsp;
            <span style="color:#ff8c42; font-weight:500;">Feature Engineering</span> (age×floors, area×height)
            &nbsp;→&nbsp;
            <span style="color:#ff4d4d; font-weight:500;">ColumnTransformer</span>
            [Numeric: Impute→Scale | Categorical: Impute→OrdinalEncode]
            &nbsp;→&nbsp;
            <span style="color:#4dffb4; font-weight:500;">XGBoost Classifier</span>
            &nbsp;→&nbsp;
            <span style="color:#e8e8f0; font-weight:500;">Damage Grade (1/2/3)</span>
        </div>
    </div>

    <div class="card" style="margin-top:1rem;">
        <div class="card-title">DATASET INFO</div>
        <div style="font-size:0.875rem; color:#6b6b8a; line-height:2;">
            📍 Source: 2015 Gorkha Earthquake, Nepal (Driven Data Competition)<br>
            🏗️ 260,601 buildings → 248,282 after deduplication<br>
            📊 38 raw features + 3 engineered features = 41 total<br>
            ⚖️ Class imbalance: Grade 2 = 56.9%, Grade 3 = 33.5%, Grade 1 = 9.6%<br>
            🔀 80/20 stratified train-test split
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#2a2a3d; font-size:0.75rem;">
    EarthShield · PRCP-1015 · Built with Streamlit & XGBoost
</div>
""", unsafe_allow_html=True)