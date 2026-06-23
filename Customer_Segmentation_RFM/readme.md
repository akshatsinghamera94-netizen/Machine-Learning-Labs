# Customer Segmentation using RFM Analysis, K-Means Clustering, and PCA

## Overview

This project performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering. The objective is to group customers based on purchasing behavior so businesses can design targeted marketing and retention strategies.

The final model is deployed using Streamlit, where users can enter customer information and identify the corresponding customer segment.

---

## Problem Statement

Businesses often treat all customers similarly, despite significant differences in purchasing behavior.

The goal of this project is to identify meaningful customer groups based on:

* Recency: How recently a customer made a purchase
* Frequency: How often a customer purchases
* Monetary: How much a customer spends

These segments can be used for personalized marketing and customer retention.

---

## Dataset

**Online Retail Dataset**

The dataset contains transaction records from a UK-based online retailer.

### Features Used

* CustomerID
* InvoiceNo
* InvoiceDate
* Quantity
* UnitPrice

---

## Data Preprocessing

* Removed missing Customer IDs
* Removed cancelled transactions
* Removed invalid quantities and prices
* Created Total Purchase Amount feature
* Performed RFM feature engineering
* Applied Log Transformation to reduce skewness
* Applied StandardScaler before clustering

---

## RFM Analysis

Three customer behavior metrics were created:

* Recency → Days since last purchase
* Frequency → Number of unique purchases
* Monetary → Total amount spent

Each customer was represented using these three features.

---

## Clustering

K-Means clustering was applied on the scaled RFM features.

The optimal number of clusters was determined using:

* Elbow Method
* Silhouette Analysis

Final number of clusters:

**K = 4**

---

## Customer Segments

### VIP Customers

* High spending
* Frequent purchases
* Recent activity

### Loyal Customers

* Consistent purchasing behavior
* Good spending patterns

### Regular Customers

* Moderate engagement
* Growth potential

### At-Risk Customers

* Low spending
* Inactive for a long period

---

## PCA Visualization

Principal Component Analysis (PCA) was used to reduce dimensions and visualize customer clusters in two dimensions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* K-Means Clustering
* PCA
* Streamlit
* Joblib
* Matplotlib
* Seaborn

---

## Project Structure

Customer_Segmentation_RFM/

├── app.py

├── kmeans.pkl

├── scaler.pkl

├── customer_segments.csv

├── Customer_Segmentation_RFM.ipynb

├── requirements.txt

└── README.md

---

## Results

The model successfully segmented 4,338 customers into four meaningful business groups:

* VIP Customers
* Loyal Customers
* Regular Customers
* At-Risk Customers

These segments can support targeted marketing campaigns and improve customer retention strategies.

---

## Author

Akshat Singh Amera
