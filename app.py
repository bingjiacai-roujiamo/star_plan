import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os
import base64
from io import BytesIO
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Step 1: Ensure reduced scaler exists
# ---------------------------
def ensure_reduced_scaler():
    full_scaler_path = "model/scaler.pkl"
    reduced_scaler_path = "model/scaler_reduced.pkl"
    selected_features = ['HBsAgbaseline', 'ALT12w_HBsAg12w', 'ALT12w_HBsAg', 'HBsAb12w', 'DNA12w']

    if not os.path.exists(reduced_scaler_path):
        scaler_full = joblib.load(full_scaler_path)
        selected_indices = [list(scaler_full.feature_names_in_).index(f) for f in selected_features]

        scaler_reduced = StandardScaler()
        scaler_reduced.mean_ = scaler_full.mean_[selected_indices]
        scaler_reduced.scale_ = scaler_full.scale_[selected_indices]
        scaler_reduced.var_ = scaler_full.var_[selected_indices]
        scaler_reduced.n_features_in_ = len(selected_features)
        scaler_reduced.feature_names_in_ = np.array(selected_features, dtype=object)

        joblib.dump(scaler_reduced, reduced_scaler_path)
        print("✅ Reduced scaler created and saved.")

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(
    page_title="HBsAg Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Ensure reduced scaler exists before loading
ensure_reduced_scaler()

@st.cache_resource
def load_model_and_preprocessors():
    model_dir = "model"
    lgbm_model = joblib.load(os.path.join(model_dir, "LGB_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler_reduced.pkl"))
    final_selected_features = ['HBsAgbaseline', 'ALT12w_HBsAg12w', 'ALT12w_HBsAg',
                               'HBsAg12wdown_1', 'HBsAb12w', 'DNA12w']
    numeric_features = ['HBsAgbaseline', 'ALT12w_HBsAg12w', 'ALT12w_HBsAg', 'HBsAb12w', 'DNA12w']
    return lgbm_model, scaler, final_selected_features, numeric_features

lgbm_model, scaler, final_selected_features, numeric_features = load_model_and_preprocessors()

def prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week12_hbsab, week12_dna):
    baseline_hbsag = 0.01 if baseline_hbsag <= 0.05 else baseline_hbsag
    week12_hbsag = 0.01 if week12_hbsag <= 0.05 else week12_hbsag

    alt12w_hbsag12w = week12_alt / week12_hbsag if week12_hbsag > 0 else 0
    alt12w_hbsag = week12_alt / baseline_hbsag if baseline_hbsag > 0 else 0
    hbsag12wdown_1 = 1 if np.log10(baseline_hbsag / week12_hbsag) >= 1 else 0

    input_df = pd.DataFrame({
        'HBsAgbaseline': [baseline_hbsag],
        'ALT12w_HBsAg12w': [alt12w_hbsag12w],
        'ALT12w_HBsAg': [alt12w_hbsag],
        'HBsAg12wdown_1': [hbsag12wdown_1],
        'HBsAb12w': [week12_hbsab],
        'DNA12w': [week12_dna]
    })

    display_df = input_df.copy()

    # Standardize numeric features
    temp_df = input_df[numeric_features]
    temp_df_scaled = pd.DataFrame(
        scaler.transform(temp_df.loc[:, scaler.feature_names_in_]),
        columns=scaler.feature_names_in_,
        index=temp_df.index
    )
    input_df[numeric_features] = temp_df_scaled

    return input_df, display_df

def predict(input_df):
    return lgbm_model.predict_proba(input_df)[0, 1]

def generate_shap_explanation(input_df, display_df):
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer(input_df)

    example_shap_values = shap.Explanation(
        values=shap_values.values[0, :],
        base_values=explainer.expected_value,
        data=display_df.values[0],
        feature_names=display_df.columns.tolist()
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(example_shap_values, show=False)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_base64

# ---------------------------
# UI Section
# ---------------------------
st.title("HBsAg Clearance Prediction")
st.write("Predict the probability of hepatitis B surface antigen clearance at 48 weeks")

with st.container():
    st.subheader("Enter Patient Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        baseline_hbsag = st.number_input("Baseline HBsAg (IU/mL)", 0.0, 25000.0, 10.0, 1.0)
    with col2:
        week12_hbsag = st.number_input("Week 12 HBsAg (IU/mL)", 0.0, 25000.0, 10.0, 1.0)
        st.caption("ℹ️ Enter 0.05 if ≤ 0.05; system internally converts to 0.01")
    with col3:
        week12_alt = st.number_input("Week 12 ALT (U/L)", 0, 5000, 40, 1)

    col4, col5 = st.columns(2)
    with col4:
        week12_hbsab = st.number_input("Week 12 HBsAb (IU/L)", 0.0, value=0.0, step=0.1)
    with col5:
        week12_dna = st.number_input("Week 12 DNA (IU/ml)", 0.0, value=0.0, step=0.1)

if st.button("Predict"):
    input_df, display_df = prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week12_hbsab, week12_dna)
    prediction = predict(input_df)
    hbsag12wdown_1 = display_df['HBsAg12wdown_1'].values[0]

    st.subheader("Prediction Result")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("half year HBsAg Clearance Probability", f"{prediction:.1%}")
        if prediction < 0.3:
            st.error("Low clearance probability")
        elif prediction < 0.7:
            st.warning("Moderate clearance probability")
        else:
            st.success("High clearance probability")

    with col2:
        st.subheader("Calculated Features")
        st.table(pd.DataFrame({
            "Feature": [
                "Baseline HBsAg", "Week 12 HBsAg", "Week 12 ALT", "Week 12 HBsAb",
                "Week 12 DNA", "ALT / HBsAg12w", "ALT / HBsAgbaseline", "HBsAg decline ≥1 log"
            ],
            "Value": [
                f"{baseline_hbsag:.2f}", f"{week12_hbsag:.2f}", f"{week12_alt}",
                f"{week12_hbsab:.2f}", f"{week12_dna:.2f}",
                f"{display_df['ALT12w_HBsAg12w'].values[0]:.4f}",
                f"{display_df['ALT12w_HBsAg'].values[0]:.4f}",
                "Yes" if hbsag12wdown_1 else "No"
            ]
        }))

    st.subheader("Model Explanation (SHAP)")
    shap_image = generate_shap_explanation(input_df, display_df)
    st.markdown(f"<img src='data:image/png;base64,{shap_image}' style='width: 100%;'>", unsafe_allow_html=True)
    st.info("Red features increase clearance probability, blue features decrease it, bar width indicates impact strength")

st.markdown("---")
st.caption("© 2025 - HBV Clearance Prediction Tool")
