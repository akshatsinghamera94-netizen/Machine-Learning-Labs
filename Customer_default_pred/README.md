# Customer Default Prediction

## Overview

This project predicts whether a credit card customer is likely to default on their payment in the next month using Machine Learning.

The goal is to help financial institutions identify high-risk customers and make better credit-related decisions.

---

## Problem Statement

Credit card default can lead to significant financial losses for banks and financial organizations. By analyzing customer demographics, repayment history, billing information, and payment behavior, we can build a model to estimate default risk.

---

## Dataset

**UCI Credit Card Default Dataset**

The dataset contains information about:

* Customer demographics
* Credit limit
* Repayment history
* Bill statements
* Previous payments

Target Variable:

* `default.payment.next.month`

  * 0 → No Default
  * 1 → Default

---

## Data Preprocessing

* Removed unnecessary columns
* Performed Exploratory Data Analysis (EDA)
* Created new features:

  * `TOTAL_BILL`
  * `TOTAL_PAYMENT`
  * `PAYMENT_RATIO`
* Split data into training and testing sets

---

## Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

After comparing multiple models, **Gradient Boosting** achieved the best overall performance.

---

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 81.87% |
| Precision | 66.62% |
| Recall    | 36.10% |
| F1 Score  | 46.82% |
| ROC-AUC   | 0.781  |

---

## Feature Importance

The most influential features were:

* PAY_0
* PAY_2
* PAYMENT_RATIO
* PAY_3
* LIMIT_BAL

These features had the strongest impact on predicting customer default.

---

## Deployment

The trained model was deployed using **Streamlit**.

Users can enter customer information and receive:

* Default Probability
* Risk Assessment (High Risk / Low Risk)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Project Structure

Customer_Default_Prediction/

├── app.py

├── model.pkl

├── requirements.txt

├── README.md

└── Customer_Default_Pred.ipynb

---

## Results

The final Gradient Boosting model successfully identified customers at risk of default and achieved strong classification performance on unseen data.

This project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model comparison, evaluation, and deployment.

---

## Author

Akshat Singh Amera
