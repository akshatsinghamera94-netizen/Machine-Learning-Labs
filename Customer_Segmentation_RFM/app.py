import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="centered"
)

# =====================================
# Load Saved Objects
# =====================================

@st.cache_resource
def load_artifacts():

    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")

    return scaler, kmeans


scaler, kmeans = load_artifacts()

# =====================================
# Cluster Names
# =====================================

cluster_names = {

    0: "🆕 Regular Customer",

    1: "⭐ VIP Customer",

    2: "👍 Loyal Customer",

    3: "⚠️ At-Risk Customer"
}

# =====================================
# App Title
# =====================================

st.title("🛒 Customer Segmentation using RFM Analysis")

st.write(
    """
Enter customer information and identify which customer segment they belong to.
"""
)

# =====================================
# User Inputs
# =====================================

recency = st.number_input(
    "Recency (Days Since Last Purchase)",
    min_value=1,
    value=30
)

frequency = st.number_input(
    "Frequency (Number of Purchases)",
    min_value=1,
    value=5
)

monetary = st.number_input(
    "Monetary Value (Total Spending)",
    min_value=1.0,
    value=1000.0
)

# =====================================
# Prediction
# =====================================

if st.button("Predict Segment"):

    # Create DataFrame

    input_df = pd.DataFrame({

        "Recency": [recency],

        "Frequency": [frequency],

        "Monetary": [monetary]

    })

    # Apply same transformation used during training

    input_log = np.log1p(input_df)

    # Scale features

    input_scaled = scaler.transform(input_log)

    # Predict cluster

    cluster = kmeans.predict(input_scaled)[0]

    # Get segment name

    segment = cluster_names[cluster]

    # Display Result

    st.success(
        f"Customer belongs to: {segment}"
    )

    # Business Interpretation

    if cluster == 1:

        st.info(
            """
            High-value customer.

            Recommended Actions:
            - Premium offers
            - Loyalty rewards
            - Exclusive membership plans
            """
        )

    elif cluster == 2:

        st.info(
            """
            Loyal customer.

            Recommended Actions:
            - Cross-sell products
            - Upsell premium plans
            - Personalized recommendations
            """
        )

    elif cluster == 0:

        st.info(
            """
            Regular customer.

            Recommended Actions:
            - Welcome offers
            - Engagement campaigns
            - Product recommendations
            """
        )

    else:

        st.info(
            """
            At-risk customer.

            Recommended Actions:
            - Re-engagement campaigns
            - Discount coupons
            - Retention offers
            """
        )

    st.subheader("Customer Data")

    st.dataframe(
        input_df,
        use_container_width=True
    )