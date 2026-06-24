import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Customer Default Prediction",
    page_icon="💳",
    layout="wide"
)

# =====================================
# Load Model
# =====================================

@st.cache_resource
def load_model():

    BASE_DIR = Path(__file__).parent

    return joblib.load(
        BASE_DIR / "model.pkl"
    )

model = load_model()

# =====================================
# Title
# =====================================

st.title("💳 Customer Default Prediction")

st.markdown(
    """
Predict whether a customer is likely to default on their credit card payment next month.
"""
)

# =====================================
# Input Form
# =====================================

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # ---------------------
    # Customer Details
    # ---------------------

    with col1:

        st.subheader("Customer Details")

        limit_bal = st.number_input(
            "Credit Limit",
            min_value=0.0,
            value=50000.0
        )

        sex = st.selectbox(
            "Gender",
            [1, 2],
            help="1 = Male, 2 = Female"
        )

        education = st.selectbox(
            "Education",
            [1, 2, 3, 4],
            help="""
1 = Graduate School
2 = University
3 = High School
4 = Others
"""
        )

        marriage = st.selectbox(
            "Marriage",
            [1, 2, 3],
            help="""
1 = Married
2 = Single
3 = Others
"""
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

    # ---------------------
    # Repayment History
    # ---------------------

    with col2:

        st.subheader("Repayment Status")

        pay_0 = st.number_input("PAY_0", value=0)
        pay_2 = st.number_input("PAY_2", value=0)
        pay_3 = st.number_input("PAY_3", value=0)
        pay_4 = st.number_input("PAY_4", value=0)
        pay_5 = st.number_input("PAY_5", value=0)
        pay_6 = st.number_input("PAY_6", value=0)

    # ---------------------
    # Billing and Payments
    # ---------------------

    with col3:

        st.subheader("Billing Information")

        bill_amt1 = st.number_input("BILL_AMT1", value=0.0)
        bill_amt2 = st.number_input("BILL_AMT2", value=0.0)
        bill_amt3 = st.number_input("BILL_AMT3", value=0.0)
        bill_amt4 = st.number_input("BILL_AMT4", value=0.0)
        bill_amt5 = st.number_input("BILL_AMT5", value=0.0)
        bill_amt6 = st.number_input("BILL_AMT6", value=0.0)

        pay_amt1 = st.number_input("PAY_AMT1", value=0.0)
        pay_amt2 = st.number_input("PAY_AMT2", value=0.0)
        pay_amt3 = st.number_input("PAY_AMT3", value=0.0)
        pay_amt4 = st.number_input("PAY_AMT4", value=0.0)
        pay_amt5 = st.number_input("PAY_AMT5", value=0.0)
        pay_amt6 = st.number_input("PAY_AMT6", value=0.0)

    submit = st.form_submit_button(
        "Predict Default Risk"
    )

# =====================================
# Prediction
# =====================================

if submit:

    total_bill = (
        bill_amt1 + bill_amt2 + bill_amt3 +
        bill_amt4 + bill_amt5 + bill_amt6
    )

    total_payment = (
        pay_amt1 + pay_amt2 + pay_amt3 +
        pay_amt4 + pay_amt5 + pay_amt6
    )

    payment_ratio = (
        total_payment / (total_bill + 1)
    )

    input_df = pd.DataFrame({

        "LIMIT_BAL":[limit_bal],
        "SEX":[sex],
        "EDUCATION":[education],
        "MARRIAGE":[marriage],
        "AGE":[age],

        "PAY_0":[pay_0],
        "PAY_2":[pay_2],
        "PAY_3":[pay_3],
        "PAY_4":[pay_4],
        "PAY_5":[pay_5],
        "PAY_6":[pay_6],

        "BILL_AMT1":[bill_amt1],
        "BILL_AMT2":[bill_amt2],
        "BILL_AMT3":[bill_amt3],
        "BILL_AMT4":[bill_amt4],
        "BILL_AMT5":[bill_amt5],
        "BILL_AMT6":[bill_amt6],

        "PAY_AMT1":[pay_amt1],
        "PAY_AMT2":[pay_amt2],
        "PAY_AMT3":[pay_amt3],
        "PAY_AMT4":[pay_amt4],
        "PAY_AMT5":[pay_amt5],
        "PAY_AMT6":[pay_amt6],

        "TOTAL_BILL":[total_bill],
        "TOTAL_PAYMENT":[total_payment],
        "PAYMENT_RATIO":[payment_ratio]

    })

    probability = model.predict_proba(
        input_df
    )[0][1]

    prediction = int(
        probability >= 0.5
    )

    st.markdown("---")

    st.subheader(
        "Prediction Result"
    )

    st.metric(
        "Default Probability",
        f"{probability:.2%}"
    )

    if prediction == 1:

        st.error(
            " High Risk: Customer is likely to default."
        )

    else:

        st.success(
            " Low Risk: Customer is unlikely to default."
        )

    st.markdown("### Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )