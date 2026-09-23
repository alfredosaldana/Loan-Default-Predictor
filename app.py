"""
=============================================================
  Loan Default Predictor — Streamlit Application
  Home Credit Dataset
=============================================================

Run with:
    streamlit run app.py

Requirements:
    pip install streamlit joblib scikit-learn xgboost shap matplotlib pandas numpy

This app loads the trained model saved by the Jupyter notebook
and lets users enter applicant data to get a real-time default prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
#  LOAD MODEL ARTIFACTS (cached so they only load once)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, and feature names from disk."""
    model         = joblib.load("models/final_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def risk_label(prob):
    """Convert probability to human-readable risk category."""
    if prob < 0.20:
        return "🟢 LOW", "green"
    elif prob < 0.50:
        return "🟡 MEDIUM", "orange"
    else:
        return "🔴 HIGH", "red"


def make_gauge(prob):
    """Create a simple gauge chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(polar=False))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background bar
    ax.barh(0.5, 1.0, height=0.3, color="#e0e0e0", left=0, align="center")

    # Risk bar
    color = "#27ae60" if prob < 0.2 else "#f39c12" if prob < 0.5 else "#e74c3c"
    ax.barh(0.5, prob, height=0.3, color=color, left=0, align="center")

    # Text
    ax.text(prob, 0.5, f"{prob:.1%}", va="center", ha="center",
            fontsize=14, fontweight="bold", color="white")
    ax.text(0.5, 0.85, "Default Probability", va="center", ha="center",
            fontsize=10, color="#555555")

    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    plt.tight_layout()
    return fig


def preprocess_input(inputs: dict, feature_names: list, scaler) -> np.ndarray:
    """
    Convert the user's form inputs into a scaled feature vector
    that matches exactly what the model expects.
    """
    # Start with a zero vector for all features
    row = pd.Series(0.0, index=feature_names)

    # Assign numerical inputs
    numerical_map = {
        "AMT_INCOME_TOTAL" : inputs["income"],
        "AMT_CREDIT"       : inputs["credit"],
        "AMT_ANNUITY"      : inputs["annuity"],
        "DAYS_BIRTH"       : -inputs["age"] * 365,
        "DAYS_EMPLOYED"    : -inputs["employed_years"] * 365,
        "EXT_SOURCE_1"     : inputs["ext1"],
        "EXT_SOURCE_2"     : inputs["ext2"],
        "EXT_SOURCE_3"     : inputs["ext3"],
        "CNT_CHILDREN"     : inputs["children"],
        "REGION_RATING_CLIENT": inputs["region_rating"],
    }
    for col, val in numerical_map.items():
        if col in row.index:
            row[col] = val

    # Engineered features
    row["CREDIT_INCOME_RATIO"]  = inputs["credit"] / (inputs["income"] + 1)
    row["ANNUITY_INCOME_RATIO"] = inputs["annuity"] / (inputs["income"] + 1)

    # One-hot encoded categorical features
    # (The column names were created by pd.get_dummies during training)
    ohe_map = {
        f"CODE_GENDER_{inputs['gender']}"                : 1,
        f"NAME_EDUCATION_TYPE_{inputs['education']}"     : 1,
        f"NAME_INCOME_TYPE_{inputs['income_type']}"      : 1,
        f"NAME_FAMILY_STATUS_{inputs['family_status']}"  : 1,
        f"FLAG_OWN_CAR_{inputs['own_car']}"              : 1,
        f"FLAG_OWN_REALTY_{inputs['own_realty']}"        : 1,
    }
    for col, val in ohe_map.items():
        if col in row.index:
            row[col] = val

    # Scale and return as 2D array
    return scaler.transform(row.values.reshape(1, -1))


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
st.title("🏦 Loan Default Predictor")
st.markdown(
    "**Credit Risk Assessment Tool** — Enter applicant details to get a real-time default prediction."
)
st.markdown("---")

if not model_loaded:
    st.error(
        "⚠️ Model files not found. Please run the Jupyter notebook first to train and save the model.\n\n"
        "Expected files:\n- `models/final_model.pkl`\n- `models/scaler.pkl`\n- `models/feature_names.pkl`"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
#  SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────
st.sidebar.header("📋 Applicant Information")
st.sidebar.markdown("Fill in the applicant's details below:")

with st.sidebar:
    st.subheader("💰 Financial Information")
    income = st.number_input("Annual Income ($)", min_value=10_000, max_value=1_000_000,
                              value=150_000, step=5_000)
    credit = st.number_input("Loan Amount ($)", min_value=10_000, max_value=4_000_000,
                              value=450_000, step=10_000)
    annuity = st.number_input("Annual Annuity ($)", min_value=1_000, max_value=300_000,
                               value=25_000, step=1_000)

    st.subheader("👤 Demographics")
    age = st.slider("Age (years)", min_value=18, max_value=70, value=35)
    gender = st.selectbox("Gender", ["M", "F"])
    education = st.selectbox("Education Type", [
        "Higher education",
        "Secondary / secondary special",
        "Incomplete higher",
        "Lower secondary",
        "Academic degree"
    ])
    income_type = st.selectbox("Income Type", [
        "Working", "Commercial associate", "Pensioner",
        "State servant", "Unemployed", "Student", "Businessman"
    ])
    family_status = st.selectbox("Family Status", [
        "Married", "Single / not married", "Civil marriage",
        "Separated", "Widow"
    ])

    st.subheader("🏠 Housing & Assets")
    own_car = st.selectbox("Owns a Car?", ["Y", "N"])
    own_realty = st.selectbox("Owns Realty?", ["Y", "N"])

    st.subheader("💼 Employment")
    employed_years = st.slider("Years Employed", min_value=0, max_value=40, value=5)

    st.subheader("📊 Credit Bureau Scores (0–1)")
    ext1 = st.slider("External Source Score 1", 0.0, 1.0, 0.50, 0.01)
    ext2 = st.slider("External Source Score 2", 0.0, 1.0, 0.55, 0.01)
    ext3 = st.slider("External Source Score 3", 0.0, 1.0, 0.50, 0.01)

    st.subheader("👨‍👩‍👧 Other")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    region_rating = st.selectbox("Region Rating (1=best, 3=worst)", [1, 2, 3], index=1)

    predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────
#  MAIN PANEL — RESULTS
# ─────────────────────────────────────────────────────────────

# Build derived metrics for display
credit_income_ratio  = credit / (income + 1)
annuity_income_ratio = annuity / (income + 1)
avg_ext_score        = np.mean([ext1, ext2, ext3])

# Always show applicant summary
col1, col2, col3, col4 = st.columns(4)
col1.metric("Annual Income",       f"${income:,.0f}")
col2.metric("Loan Amount",         f"${credit:,.0f}")
col3.metric("Credit/Income Ratio", f"{credit_income_ratio:.2f}x")
col4.metric("Avg Credit Score",    f"{avg_ext_score:.3f}")

st.markdown("---")

if predict_btn:
    # ── Prepare inputs ──────────────────────────────────────
    inputs = {
        "income": income, "credit": credit, "annuity": annuity,
        "age": age, "gender": gender, "education": education,
        "income_type": income_type, "family_status": family_status,
        "own_car": own_car, "own_realty": own_realty,
        "employed_years": employed_years, "ext1": ext1,
        "ext2": ext2, "ext3": ext3,
        "children": children, "region_rating": region_rating
    }

    X_input = preprocess_input(inputs, feature_names, scaler)
    prob    = float(model.predict_proba(X_input)[0, 1])
    label, color = risk_label(prob)

    # ── Results section ──────────────────────────────────────
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("📈 Prediction Result")
        st.markdown(f"### Risk Level: {label}")
        gauge_fig = make_gauge(prob)
        st.pyplot(gauge_fig, use_container_width=True)
        st.caption("⚠️ This prediction is for educational purposes only.")

    with res_col2:
        st.subheader("🔍 Key Risk Indicators")

        indicators = {
            "Credit-to-Income Ratio"   : (f"{credit_income_ratio:.2f}x",
                                          "⚠️ High" if credit_income_ratio > 6 else "✅ OK"),
            "Annuity-to-Income Ratio"  : (f"{annuity_income_ratio:.2%}",
                                          "⚠️ High" if annuity_income_ratio > 0.4 else "✅ OK"),
            "Avg External Credit Score": (f"{avg_ext_score:.3f}",
                                          "✅ Good" if avg_ext_score > 0.5 else "⚠️ Low"),
            "Employment Length"        : (f"{employed_years} years",
                                          "⚠️ Short" if employed_years < 2 else "✅ Stable"),
            "Age"                      : (f"{age} years",
                                          "⚠️ Young" if age < 25 else "✅ OK"),
        }

        for indicator, (value, status) in indicators.items():
            i1, i2, i3 = st.columns([2, 1, 1])
            i1.write(indicator)
            i2.write(value)
            i3.write(status)

    # ── SHAP Explanation ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 Why This Prediction? (SHAP Explanation)")
    st.markdown(
        "The chart below shows which features **increased** (red) or **decreased** (blue) "
        "this applicant's default probability."
    )

    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_input)

        shap_df = pd.DataFrame({
            "Feature" : feature_names,
            "SHAP"    : shap_vals[0]
        }).reindex(pd.Series(shap_vals[0]).abs().sort_values(ascending=False).index)
        top_shap = shap_df.head(10)

        fig2, ax2 = plt.subplots(figsize=(9, 5))
        colors_s = ["#e74c3c" if v > 0 else "#2980b9" for v in top_shap["SHAP"]]
        ax2.barh(top_shap["Feature"], top_shap["SHAP"], color=colors_s, edgecolor="black")
        ax2.axvline(0, color="black", linewidth=0.8)
        ax2.set_title("Top 10 Feature Contributions (SHAP)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("SHAP Value (positive = increases default risk)")
        plt.tight_layout()
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"SHAP explanation could not be generated: {e}")

    # ── Decision recommendation ──────────────────────────────
    st.markdown("---")
    st.subheader("📋 Risk Summary")
    if prob < 0.20:
        st.success(
            f"**LOW RISK** — Predicted default probability: {prob:.1%}\n\n"
            "This applicant shows a low risk profile. Standard loan approval process recommended."
        )
    elif prob < 0.50:
        st.warning(
            f"**MEDIUM RISK** — Predicted default probability: {prob:.1%}\n\n"
            "This applicant shows a moderate risk profile. Additional documentation or "
            "a smaller loan amount may be advisable."
        )
    else:
        st.error(
            f"**HIGH RISK** — Predicted default probability: {prob:.1%}\n\n"
            "This applicant shows a high risk of default. Manual review by a credit officer "
            "is strongly recommended before approval."
        )

else:
    # Instructions when no prediction has been made yet
    st.info(
        "👈 Fill in the applicant's details in the sidebar and click **Predict Default Risk** to see results."
    )

    st.subheader("📖 How This App Works")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 1. Enter Data")
        st.markdown(
            "Use the sidebar to input the applicant's financial details, demographics, "
            "and external credit scores."
        )

    with col_b:
        st.markdown("### 2. Get Prediction")
        st.markdown(
            "Our XGBoost model — trained on 307,511 real loan applications — "
            "calculates the probability of default."
        )

    with col_c:
        st.markdown("### 3. Understand Why")
        st.markdown(
            "SHAP values explain exactly which factors drove the prediction, "
            "enabling transparent, fair lending decisions."
        )

    # Show overview metrics from saved predictions (if available)
    try:
        preds_df = pd.read_csv("dashboard/data_exports/predictions.csv")
        st.markdown("---")
        st.subheader("📊 Model Performance Overview (Test Set)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Samples",    f"{len(preds_df):,}")
        m2.metric("Default Rate",    f"{preds_df['ACTUAL'].mean():.1%}")
        m3.metric("High Risk Flagged",
                  f"{(preds_df['RISK_LEVEL']=='High').mean():.1%}")
        m4.metric("Low Risk Applicants",
                  f"{(preds_df['RISK_LEVEL']=='Low').mean():.1%}")
    except FileNotFoundError:
        pass

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🏦 Loan Default Predictor · Built with XGBoost + SHAP · "
    "Data: Home Credit Default Risk (Kaggle) · "
    "⚠️ Educational purposes only — not for actual lending decisions."
)
