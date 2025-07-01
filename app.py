import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Hepatitis B Surface Antigen Clearance Prediction",
    page_icon="🧬",
    layout="wide"
)

# Function to load model and preprocessing objects
@st.cache_resource
def load_model_and_preprocessors():
    model_dir = "model"
    # Load model and preprocessing objects
    lgbm_model = joblib.load(os.path.join(model_dir, "LGB_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    final_selected_features = ['HBsAgbaseline', 'ALT12w_HBsAg12w', 'ALT12w_HBsAg', 'HBsAg12wdown_1', 'HBsAb12w', 'DNA12w']
    numeric_features = final_selected_features  # All features are numeric
    categorical_features = []
    return lgbm_model, scaler, final_selected_features, numeric_features, categorical_features

# Load model and preprocessors
lgbm_model, scaler, final_selected_features, numeric_features, categorical_features = load_model_and_preprocessors()

# Function to prepare input data for prediction
def prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week12_hbsab, week12_dna):
    # Adjust HBsAg values if ≤ 0.05
    baseline_hbsag = 0.01 if baseline_hbsag <= 0.05 else baseline_hbsag
    week12_hbsag = 0.01 if week12_hbsag <= 0.05 else week12_hbsag
    
    # Calculate derived features
    hbsag_baseline = baseline_hbsag
    alt12w_hbsag12w = week12_alt / week12_hbsag if week12_hbsag > 0 else 0
    alt12w_hbsag = week12_alt / baseline_hbsag if baseline_hbsag > 0 else 0
    if baseline_hbsag > 0 and week12_hbsag > 0:
        log_diff = np.log10(baseline_hbsag / week12_hbsag)
        hbsag12wdown_1 = 1 if log_diff >= 1 else 0
    else:
        hbsag12wdown_1 = 0

    hbsab12w = week12_hbsab
    dna12w = week12_dna
    
    # Create input dataframe
    input_df = pd.DataFrame({
        'HBsAgbaseline': [hbsag_baseline],
        'ALT12w_HBsAg12w': [alt12w_hbsag12w],
        'ALT12w_HBsAg': [alt12w_hbsag],
        'HBsAg12wdown_1': [hbsag12wdown_1],
        'HBsAb12w': [hbsab12w],
        'DNA12w': [dna12w]
    })
    
    # Create a copy for display (unscaled)
    display_df = input_df.copy()
    
    # Scale the input dataframe
    input_df_scaled = pd.DataFrame(
        scaler.transform(input_df),
        index=input_df.index,
        columns=input_df.columns
    )
    
    return input_df_scaled, display_df

# Function to make prediction
def predict(input_df):
    # Get prediction probability
    pred_proba = lgbm_model.predict_proba(input_df)[0, 1]
    return pred_proba

# Function to generate SHAP explanation
def generate_shap_explanation(input_df, display_df):
    # Create the explainer
    explainer = shap.TreeExplainer(lgbm_model)
    
    # Calculate SHAP values
    shap_values = explainer(input_df)
    
    # Use unscaled values for visualization
    display_values = display_df.values[0]
    feature_names = list(input_df.columns)
    
    # Create a SHAP explanation for the waterfall plot
    example_shap_values = shap.Explanation(
        values=shap_values.values[0, :, 1],
        base_values=explainer.expected_value[1],
        data=display_values,
        feature_names=feature_names
    )
    
    # Create the waterfall plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(example_shap_values, show=False)
    plt.title("Feature Impact on Prediction", fontsize=14)
    plt.tight_layout()
    
    # Convert the plot to a base64 encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return image_base64

# UI
st.title("Hepatitis B Surface Antigen Clearance Prediction")
st.write("This tool predicts the probability of hepatitis B surface antigen clearance at 48 weeks based on baseline and 12-week measurements.")

with st.container():
    st.subheader("Patient Measurements")
    col1, col2, col3 = st.columns(3)
    with col1:
        baseline_hbsag = st.number_input("Baseline HBsAg (IU/mL)", 
                                         min_value=0.0, 
                                         max_value=25000.0, 
                                         value=10.0, 
                                         step=1.0)
    with col2:
        week12_hbsag = st.number_input("Week 12 HBsAg (IU/mL)", 
                                       min_value=0.0, 
                                       max_value=25000.0, 
                                       value=10.0, 
                                       step=1.0)
        st.caption("ℹ️ Enter 0.05 if ≤ 0.05; adjusted to 0.01.")
    with col3:
        week12_alt = st.number_input("Week 12 ALT (IU/L)", 
                                     min_value=0, 
                                     max_value=5000, 
                                     value=40, 
                                     step=1)
    col4, col5 = st.columns(2)
    with col4:
        week12_hbsab = st.number_input("Week 12 HBsAb", 
                                       min_value=0.0, 
                                       value=0.0, 
                                       step=0.1)
    with col5:
        week12_dna = st.number_input("Week 12 DNA", 
                                     min_value=0.0, 
                                     value=0.0, 
                                     step=0.1)

if st.button("Calculate Prediction"):
    # Prepare input data
    input_df, display_df = prepare_input_data(baseline_hbsag, week12_hbsag, week12_alt, week12_hbsab, week12_dna)
    
    # Extract calculated features for display
    alt12w_hbsag12w = display_df['ALT12w_HBsAg12w'].values[0]
    alt12w_hbsag = display_df['ALT12w_HBsAg'].values[0]
    hbsag12wdown_1 = display_df['HBsAg12wdown_1'].values[0]
    
    # Make prediction
    prediction = predict(input_df)
    
    # Display results
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Probability of HBsAg Clearance at 48 Weeks", f"{prediction:.1%}")
        if prediction < 0.3:
            st.error("Low probability of HBsAg clearance")
        elif prediction < 0.7:
            st.warning("Moderate probability of HBsAg clearance")
        else:
            st.success("High probability of HBsAg clearance")
    
    with col2:
        st.subheader("Calculated Features")
        feature_data = {
            "Feature": [
                "Baseline HBsAg (IU/mL)",
                "Week 12 HBsAg (IU/mL)",
                "Week 12 ALT (IU/L)",
                "Week 12 HBsAb",
                "Week 12 DNA",
                "ALT12w / HBsAg12w ratio",
                "ALT12w / HBsAgbaseline ratio",
                "HBsAg decline ≥ 10 IU/mL"
            ],
            "Value": [
                f"{baseline_hbsag:.2f}",
                f"{week12_hbsag:.2f}",
                f"{week12_alt}",
                f"{week12_hbsab:.2f}",
                f"{week12_dna:.2f}",
                f"{alt12w_hbsag12w:.4f}",
                f"{alt12w_hbsag:.4f}",
                "Yes" if hbsag12wdown_1 == 1 else "No"
            ]
        }
        st.table(pd.DataFrame(feature_data))
    
    # Generate and display SHAP explanation
    shap_image = generate_shap_explanation(input_df, display_df)
    st.subheader("Model Explanation (SHAP Values)")
    st.markdown(f"<img src='data:image/png;base64,{shap_image}' style='width: 100%;'>", unsafe_allow_html=True)
    
    st.info("""
    **Interpretation Guide:**
    - Features in red push the prediction toward HBsAg clearance (higher probability)
    - Features in blue push the prediction away from HBsAg clearance (lower probability)
    - The width of each bar shows how strongly that feature affects the prediction
    """)
    
    st.caption("Note: This is a prediction model and should be used as a tool to aid clinical decision-making, not as a replacement for clinical judgment.")

# Footer
st.markdown("---")
st.caption("© 2025 - HBV Clearance Prediction Tool")
