import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Earthquake Building Damage Predictor",
    page_icon="🏚️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("earthquake_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    bundle = load_model()
    model = bundle["model"]
    features = bundle["features"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏚️ Earthquake Damage Predictor")
st.sidebar.markdown(
    "Predict building damage grade from the **2015 Gorkha earthquake** dataset.\n\n"
    "**Damage grades:**\n- 🟢 1 = Low damage\n- 🟡 2 = Medium damage\n- 🔴 3 = Almost complete destruction"
)

page = st.sidebar.radio("Navigate", ["Single Prediction", "Batch Prediction (CSV)", "Model Info"])

# ── Helpers ───────────────────────────────────────────────────────────────────
DAMAGE_LABELS = {1: "🟢 Low Damage", 2: "🟡 Medium Damage", 3: "🔴 Almost Complete Destruction"}
DAMAGE_COLORS = {1: "green", 2: "orange", 3: "red"}

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

def build_input_df(vals: dict) -> pd.DataFrame:
    """Return a single-row DataFrame matching the training feature order."""
    return pd.DataFrame([vals])[features]


def target_encode_geo(df: pd.DataFrame, geo_means: dict) -> pd.DataFrame:
    """Apply target encoding to geo columns using pre-computed means."""
    df = df.copy()
    for col, mean_map in geo_means.items():
        if col in df.columns:
            df[col] = df[col].map(mean_map).fillna(mean_map.get("__global__", 2.0))
    return df


# ── Page: Single Prediction ───────────────────────────────────────────────────
if page == "Single Prediction":
    st.title("🏚️ Building Damage Prediction")
    st.markdown("Fill in the building characteristics below and click **Predict**.")

    if not MODEL_LOADED:
        st.error("⚠️ `earthquake_model.pkl` not found. Place the saved model file in the same directory as `app.py`.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Geographic")
        geo1 = st.number_input("Geo Level 1 ID (target-encoded mean)", 0.0, 3.0, 2.0, 0.01)
        geo2 = st.number_input("Geo Level 2 ID (target-encoded mean)", 0.0, 3.0, 2.0, 0.01)
        geo3 = st.number_input("Geo Level 3 ID (target-encoded mean)", 0.0, 3.0, 2.0, 0.01)

        st.subheader("Building Basics")
        floors      = st.slider("Floors before earthquake", 1, 10, 2)
        age         = st.slider("Building age (years)", 0, 200, 20)
        area_pct    = st.slider("Area percentage", 1, 100, 10)
        height_pct  = st.slider("Height percentage", 1, 100, 5)
        families    = st.slider("Number of families", 0, 20, 1)

    with col2:
        st.subheader("Structural Features")
        land   = st.selectbox("Land surface condition",  CAT_OPTIONS["land_surface_condition"])
        found  = st.selectbox("Foundation type",         CAT_OPTIONS["foundation_type"])
        roof   = st.selectbox("Roof type",               CAT_OPTIONS["roof_type"])
        gfloor = st.selectbox("Ground floor type",       CAT_OPTIONS["ground_floor_type"])
        ofloor = st.selectbox("Other floor type",        CAT_OPTIONS["other_floor_type"])
        pos    = st.selectbox("Position",                CAT_OPTIONS["position"])
        plan   = st.selectbox("Plan configuration",      CAT_OPTIONS["plan_configuration"])
        legal  = st.selectbox("Legal ownership status",  CAT_OPTIONS["legal_ownership_status"])

    with col3:
        st.subheader("Superstructure Materials")
        sup_cols = [c for c in features if c.startswith("has_superstructure")]
        sup_vals = {}
        for col in sup_cols:
            label = col.replace("has_superstructure_", "").replace("_", " ").title()
            sup_vals[col] = int(st.checkbox(label, value=False))

        st.subheader("Secondary Use")
        use_cols = [c for c in features if c.startswith("has_secondary_use")]
        use_vals = {}
        for col in use_cols:
            label = col.replace("has_secondary_use_", "").replace("_", " ").title()
            use_vals[col] = int(st.checkbox(label, value=False, key=col))

    if st.button("🔍 Predict Damage Grade", type="primary"):
        vals = {
            "geo_level_1_id":       geo1,
            "geo_level_2_id":       geo2,
            "geo_level_3_id":       geo3,
            "count_floors_pre_eq":  floors,
            "age":                  age,
            "area_percentage":      area_pct,
            "height_percentage":    height_pct,
            "count_families":       families,
            "land_surface_condition": land,
            "foundation_type":      found,
            "roof_type":            roof,
            "ground_floor_type":    gfloor,
            "other_floor_type":     ofloor,
            "position":             pos,
            "plan_configuration":   plan,
            "legal_ownership_status": legal,
            "age_x_floors":         age * floors,
            "area_x_height":        area_pct * height_pct,
            **sup_vals,
            **use_vals,
        }

        # Fill any missing features with 0
        for f in features:
            if f not in vals:
                vals[f] = 0

        input_df = build_input_df(vals)
        pred = model.predict(input_df)[0] + 1   # XGBoost was trained on y-1
        proba = model.predict_proba(input_df)[0]

        st.divider()
        st.subheader("Prediction Result")
        grade_label = DAMAGE_LABELS[pred]
        st.markdown(f"### Predicted Damage Grade: **{grade_label}**")

        prob_df = pd.DataFrame({
            "Damage Grade": [DAMAGE_LABELS[i] for i in [1, 2, 3]],
            "Probability":  [round(p * 100, 1) for p in proba]
        })
        st.bar_chart(prob_df.set_index("Damage Grade"))


# ── Page: Batch Prediction ────────────────────────────────────────────────────
elif page == "Batch Prediction (CSV)":
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a CSV with the same columns as the training data "
        "(**geo columns should already be target-encoded**). "
        "The app will append a `predicted_damage_grade` column."
    )

    if not MODEL_LOADED:
        st.error("⚠️ `earthquake_model.pkl` not found.")
        st.stop()

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Preview (first 5 rows):", df.head())

        # Engineer interaction features if not present
        if "age_x_floors" not in df.columns:
            df["age_x_floors"] = df["age"] * df["count_floors_pre_eq"]
        if "area_x_height" not in df.columns:
            df["area_x_height"] = df["area_percentage"] * df["height_percentage"]

        # Fill any missing feature columns with 0
        for f in features:
            if f not in df.columns:
                df[f] = 0

        preds = model.predict(df[features]) + 1
        df["predicted_damage_grade"] = preds
        df["damage_label"] = df["predicted_damage_grade"].map({
            1: "Low", 2: "Medium", 3: "Almost Complete"
        })

        st.success(f"✅ Predicted {len(df):,} buildings.")
        st.write(df[["predicted_damage_grade", "damage_label"]].value_counts().reset_index())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")


# ── Page: Model Info ──────────────────────────────────────────────────────────
elif page == "Model Info":
    st.title("ℹ️ Model Information")

    if MODEL_LOADED:
        st.subheader("Saved Model Metadata")
        info = {k: v for k, v in bundle.items() if k != "model"}
        for k, v in info.items():
            st.write(f"**{k}:** {v}")
    else:
        st.warning("Model file not loaded — showing project summary only.")

    st.subheader("Project Summary")
    st.markdown("""
| Item | Detail |
|---|---|
| Dataset | 2015 Gorkha Earthquake, Nepal — 260,601 buildings |
| Task | Multiclass ordinal classification (damage grade 1/2/3) |
| Best model | XGBoost (tuned) |
| Accuracy | 0.7529 |
| F1 Macro | 0.7046 |
| CV std | ±0.003 |

**Top features (by importance):**
1. `geo_level_3_id` (target-encoded) — 0.179
2. `roof_type` — 0.104
3. `has_superstructure_mud_mortar_stone`
4. `foundation_type`
5. `age_x_floors` (engineered feature)

**Key findings:**
- Geographic micro-region is the strongest damage predictor
- Mud mortar stone superstructures → highest damage risk
- RC engineered construction → lowest damage risk
- Older + taller buildings are most vulnerable
    """)
