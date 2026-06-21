# Customer Churn Prediction using ANN

## Overview

Customer retention is a key challenge for subscription-based businesses. This project uses an Artificial Neural Network (ANN) to predict whether a telecom customer is likely to churn based on their demographics, service usage, billing information, and account details.

The final model is deployed as an interactive web application using Streamlit.

---

## Problem Statement

The goal is to identify customers who are likely to discontinue their telecom services so that businesses can take proactive retention measures.

This is a **binary classification problem**:

* **0** → Customer stays
* **1** → Customer churns

---

## Dataset

This project uses the **IBM Telco Customer Churn Dataset** containing information about **7,043 customers**.

### Target Variable

* `Churn Value`

### Data Includes

* Customer demographics
* Subscription details
* Billing information
* Customer lifetime value (CLTV)

---

## Data Preprocessing

* Removed irrelevant and leakage-prone columns
* Converted `Total Charges` to numeric format
* Handled missing values using `SimpleImputer`
* Encoded categorical features using `OneHotEncoder`
* Scaled numerical features using `StandardScaler`
* Built a reusable preprocessing pipeline using `ColumnTransformer`

---

## Model Architecture

The ANN was built using TensorFlow and Keras with:

* Dense Layer: 64 neurons, ReLU
* Dropout: 30%
* Dense Layer: 32 neurons, ReLU
* Dropout: 20%
* Output Layer: Sigmoid activation

### Training Configuration

* Optimizer: Adam
* Loss Function: Binary Cross-Entropy
* Early Stopping: Enabled

---

## Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 79.84% |
| Precision | 64.06% |
| Recall    | 54.81% |
| F1 Score  | 59.08% |
| ROC-AUC   |  0.846 |

The model achieved a strong ROC-AUC score, demonstrating good capability in distinguishing customers likely to churn.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Joblib

---

## Project Structure

```text
Customer_Churn_ANN/
│
├── app.py
├── model.keras
├── preprocessor.pkl
├── Customer_Churn_ANN.ipynb
├── requirements.txt
└── README.md
```

---

## Deployment

The model is deployed using Streamlit, allowing users to input customer details and receive churn predictions in real time.

---

## Key Learnings

* Building end-to-end ML pipelines
* Training and evaluating ANNs
* Preventing overfitting using Dropout and Early Stopping
* Deploying machine learning models with Streamlit

---

## Author

**Akshat Singh Amera**
