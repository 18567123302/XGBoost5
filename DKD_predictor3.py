import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="DKD Risk Predictor", layout="centered")

# -----------------------------
# Load model (cache for speed)
# -----------------------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model("xgb_model.pkl")

# -----------------------------
# Feature names (must match training order)
# -----------------------------
feature_names = ["Hb", "PLT", "ALT", "BUN", "UA", "HDL"]

# -----------------------------
# UI
# -----------------------------
st.title("Risk Prediction of Diabetic Nephropathy in Elderly Patients with Type 2 Diabetes in the Community")
st.title("社区老年二型糖尿病患者糖尿病肾病风险预测")

st.markdown("---")

# Optional debug switch
debug = st.checkbox("Show debug info (显示调试信息)", value=False)

# Use a form to ensure a clean "submit to predict" workflow
with st.form("predict_form"):
    hb = st.number_input(
        "Hb (Hemoglobin) (血红蛋白) <g/L>:",
        min_value=50.0, max_value=200.0, value=120.0, step=1.0
    )

    plt_count = st.number_input(
        "PLT (Platelets) (血小板) <10^9/L>:",
        min_value=10.0, max_value=500.0, value=280.0, step=1.0
    )

    alt = st.number_input(
        "ALT (Alanine Aminotransferase) (血清谷丙转氨酶) <U/L>:",
        min_value=0.0, max_value=500.0, value=25.0, step=1.0
    )

    bun = st.number_input(
        "BUN (Blood urea nitrogen) (血尿素氮) <mmol/L>:",
        min_value=0.0, max_value=50.0, value=5.0, step=0.1
    )

    ua = st.number_input(
        "UA (Uric Acid) (尿酸) <μmol/L>:",
        min_value=100.0, max_value=800.0, value=350.0, step=1.0
    )

    hdl = st.number_input(
        "HDL (High-Density Lipoprotein Cholesterol) (高密度脂蛋白胆固醇) <mmol/L>:",
        min_value=0.1, max_value=10.0, value=2.0, step=0.1
    )

    submit = st.form_submit_button("Predict (预测)")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    feature_values = [hb, plt_count, alt, bun, ua, hdl]
    X = pd.DataFrame([feature_values], columns=feature_names)
    features = X.values.astype(float)

    # Predict class + probability
    predicted_class = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]  # [P(class0), P(class1)]

    # DKD risk probability = probability of class 1
    risk = float(proba[1]) * 100
    non_risk = float(proba[0]) * 100

    st.markdown("### Results (结果)")

    st.write(f"**Predicted Class (预测类别):** {predicted_class}")
    st.write(f"**Prediction Probabilities (预测概率):** {proba}")

    # Advice (use risk=class1 probability)
    if predicted_class == 1:
        advice = (
            "According to this model, you may have a higher risk of developing diabetic nephropathy.\n"
            f"The predicted DKD risk probability is **{risk:.1f}%**.\n"
            "It is recommended that you see a doctor as soon as possible for further evaluation and appropriate management.\n\n"
            "根据模型预测，您可能存在较高的糖尿病肾病发病风险。\n"
            f"模型预测的糖尿病肾病（DKD）风险概率为 **{risk:.1f}%**。\n"
            "建议您尽快就医，以进行进一步评估并采取适当的管理措施。"
        )
    else:
        advice = (
            "According to the model, your DKD risk is relatively low.\n"
            f"The predicted probability of **not** having DKD is **{non_risk:.1f}%** (DKD risk: **{risk:.1f}%**).\n"
            "It is recommended that you maintain a healthy lifestyle and monitor your health regularly. If you experience any symptoms, please see a doctor promptly.\n\n"
            "根据模型预测，您的糖尿病肾病风险相对较低。\n"
            f"模型预测的**无糖尿病肾病**概率为 **{non_risk:.1f}%**（DKD 风险：**{risk:.1f}%**）。\n"
            "建议您继续保持健康的生活方式，并定期监测健康状况。如有任何异常症状，请及时就医。"
        )

    st.markdown(advice)

    if debug:
        st.markdown("#### Debug info (调试信息)")
        st.write("Inputs used by model:", X)
        st.write("Predicted proba:", proba)
        st.write("DKD risk (class=1) %:", risk)

    # -----------------------------
    # SHAP force plot
    # -----------------------------
    st.markdown("---")
    st.markdown("### SHAP Explanation (SHAP 解释)")

    try:
        # Use booster for stability
        explainer = shap.TreeExplainer(model.get_booster())
        shap_values = explainer.shap_values(X)

        # Binary classification: shap_values may be list [class0, class1]
        if isinstance(shap_values, list):
            shap_vec = shap_values[1][0]  # class 1
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1]
        else:
            shap_vec = shap_values[0]
            base_value = explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0]

        plt.figure()
        shap.force_plot(base_value, shap_vec, X.iloc[0, :], matplotlib=True)
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP plot could not be generated: {e}")
        if debug:
            st.exception(e)
