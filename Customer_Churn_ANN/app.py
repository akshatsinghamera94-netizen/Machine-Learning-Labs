import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# -----------------------------
# Load Model and Preprocessor
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

model = load_model()
preprocessor = load_preprocessor()

# -----------------------------
# Title
# -----------------------------
st.title("📉 Customer Churn Prediction using ANN")
st.markdown(
    """
Predict whether a telecom customer is likely to churn based on their subscription details and usage patterns.
"""
)

# -----------------------------
# Input Form
# -----------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["Yes", "No"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure_months = st.slider(
            "Tenure Months",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=1000.0
        )

        cltv = st.number_input(
            "CLTV",
            min_value=0,
            max_value=10000,
            value=5000
        )

    submit = st.form_submit_button("Predict Churn")

# -----------------------------
# Prediction
# -----------------------------
if submit:

    input_df = pd.DataFrame({
        "Gender": [gender],
        "Senior Citizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure Months": [tenure_months],
        "Phone Service": [phone_service],
        "Multiple Lines": [multiple_lines],
        "Internet Service": [internet_service],
        "Online Security": [online_security],
        "Online Backup": [online_backup],
        "Device Protection": [device_protection],
        "Tech Support": [tech_support],
        "Streaming TV": [streaming_tv],
        "Streaming Movies": [streaming_movies],
        "Contract": [contract],
        "Paperless Billing": [paperless_billing],
        "Payment Method": [payment_method],
        "Monthly Charges": [monthly_charges],
        "Total Charges": [total_charges],
        "CLTV": [cltv]
    })

    processed_input = preprocessor.transform(input_df)

    probability = model.predict(processed_input, verbose=0)[0][0]

    prediction = int(probability >= 0.5)

    st.markdown("---")

    st.subheader("Prediction Result")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if prediction == 1:
        st.error(
            " High Risk: This customer is likely to churn."
        )

        st.info(
            "Recommended actions:\n"
            "- Offer discounts\n"
            "- Provide loyalty rewards\n"
            "- Improve customer support\n"
            "- Review contract options"
        )

    else:
        st.success(
            "✅ Low Risk: This customer is likely to stay."
        )

        st.info(
            "Recommended actions:\n"
            "- Continue engagement\n"
            "- Promote premium services\n"
            "- Encourage long-term contracts"
        )

    st.markdown("### Customer Details")

    st.dataframe(input_df, use_container_width=True)

