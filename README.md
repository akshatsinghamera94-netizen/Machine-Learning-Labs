# Machine Learning Labs

A collection of Machine Learning, Deep Learning, and Reinforcement Learning projects built using Python and modern ML frameworks.

These projects cover practical machine learning workflows, from data preprocessing and feature engineering to model development, evaluation, explainability, and deployment. The projects gradually extend from classical ML into deep learning, computer vision, and reinforcement learning for intelligent decision-making and robotics-inspired applications.

## Repository Structure

```text
Machine-Learning-Labs/

├── Customer_Default_Prediction/
├── Customer_Segmentation_RFM/
├── Customer_Churn_ANN/
├── AI_Industrial_Defect_Detection/
├── Multi_Robot_Warehouse_Fleet_Management_RL/
└── README.md

---

## Projects

### 1. Customer Default Prediction

Projects

1. Customer Default Prediction

Predicts whether a credit card customer is likely to default on their payment next month.

Key Highlights
Exploratory Data Analysis (EDA)
Data Preprocessing
Feature Engineering
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
Model Comparison
Feature Importance Analysis
Streamlit Deployment
Best Model

Gradient Boosting

Performance
Accuracy: 81.87%
ROC-AUC: 0.781


2. Customer Segmentation using RFM Analysis, K-Means, and PCA

Segments customers based on purchasing behavior using unsupervised learning techniques.

Key Highlights
RFM Feature Engineering
K-Means Clustering
Elbow Method
Silhouette Analysis
PCA Visualization
Customer Segmentation Dashboard
Customer Segments
VIP Customers
Loyal Customers
Regular Customers
At-Risk Customers


3. Customer Churn Prediction using ANN

Predicts whether a telecom customer is likely to leave the service.

Key Highlights
Data Preprocessing Pipeline
Artificial Neural Network (ANN)
TensorFlow/Keras
Dropout Regularization
Early Stopping
Model Evaluation
Streamlit Deployment
Performance
Accuracy: 79.84%
Precision: 64.06%
Recall: 54.81%
F1 Score: 59.08%
ROC-AUC: 0.846


4. AI Industrial Surface Defect Detection

A computer vision system for detecting common surface defects in hot-rolled steel using deep learning.

The project uses the NEU-DET dataset and explores both custom CNNs and transfer learning approaches.

Key Highlights
Image Data Exploration
Image Preprocessing
Custom CNN
Transfer Learning
ResNet50
EfficientNetB0
Model Comparison
Grad-CAM Explainability
Streamlit Deployment
Best Model

ResNet50

Performance
Accuracy: 99.72%
Precision: 99.73%
Recall: 99.72%
F1 Score: 99.72%


5. Multi-Robot Warehouse Fleet Management using Reinforcement Learning

A simulated warehouse fleet-management system where multiple robots are assigned tasks using a reinforcement learning policy.

The project explores task allocation, robot workload, travel distance, makespan, and comparison with a simple nearest-robot heuristic.

Key Highlights
Reinforcement Learning Environment
Gymnasium
PPO (Proximal Policy Optimization)
Multi-Robot Task Allocation
Warehouse Simulation
Nearest-Robot Baseline
Makespan Analysis
Fleet Workload Analysis
Streamlit Visualization
Evaluation

The trained PPO policy was evaluated against a nearest-robot heuristic over multiple warehouse scenarios.

Current evaluation:

Average Total Distance — Baseline: 38.48, PPO: 41.84
Average Makespan — Baseline: 15.98, PPO: 18.84

The current results show that the heuristic remains stronger for this simulation, while the project provides a foundation for exploring more advanced fleet-management and multi-agent reinforcement learning approaches.

Technologies Used
Python
Pandas
NumPy
Scikit-Learn
TensorFlow / Keras
PyTorch
Gymnasium
Stable-Baselines3
Matplotlib
Seaborn
Streamlit
Joblib
Learning Outcomes

Through these projects, I gained hands-on experience in:

Data Cleaning and Preprocessing
Exploratory Data Analysis
Feature Engineering
Supervised Learning
Unsupervised Learning
Deep Learning
Computer Vision
Transfer Learning
Model Explainability
Reinforcement Learning
Model Evaluation
Hyperparameter Tuning
Model Deployment using Streamlit
Simulation and decision-making for multi-robot systems

My current focus is on building practical machine learning systems and gradually applying these skills to intelligent decision-making, reinforcement learning, and robotics-oriented problems.

Author

Akshat Singh Amera

B.Tech – Artificial Intelligence & Data Science

Machine Learning | Deep Learning | Reinforcement Learning
