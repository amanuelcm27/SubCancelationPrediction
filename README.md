# 🔄 Customer Churn Prediction for a Subscription-Based Business

This project uses classical machine learning to predict which customers are likely to cancel their subscription to a digital service (e.g., SaaS, eLearning, streaming). By analyzing customer behavior and attributes, the model helps businesses proactively retain customers and reduce churn.

## 📌 Objective

Build a binary classification model to predict customer churn using a dataset of user activity, demographics, and service usage patterns.

## ✅ Highlights
- Cleaned and prepared data with features like tenure, monthly charges, support calls, and contract type.
- Engineered new features such as usage intensity and late payment frequency.
- Trained and compared several models: **Logistic Regression**, **Random Forest**, **XGBoost**.
- Achieved over **85% accuracy** with an optimized XGBoost model.

## 💼 Business Use Case
- Predict which users are likely to cancel next month.
- Enable customer success teams to target at-risk users with retention campaigns.
- Support dashboard insights with real ML-driven risk metrics.

## 🧪 Tools Used
- Python
- pandas, matplotlib, seaborn (EDA)
- scikit-learn
- XGBoost
- SHAP (to explain predictions)

## 📊 Model Performance
| Model             | Accuracy | Precision | Recall |
|------------------|----------|-----------|--------|
| Logistic Regression | 79.2%   | 76.8%     | 74.3%  |
| Random Forest       | 84.5%   | 82.1%     | 81.0%  |
| XGBoost             | **87.1%** | **85.3%** | **84.2%** |

## 📁 Dataset
- Public dataset: `Telco Customer Churn` (via Kaggle)
- 7,000+ customer records with 20+ features.

## 📉 Visuals
- Correlation heatmaps for feature selection
- ROC curves and confusion matrix per model
- SHAP summary plots for top predictors

## 🔍 Insights Found
- Customers with month-to-month contracts and high monthly charges are most at risk.
- Tenure (time with the company) is the strongest negative indicator of churn.
- Adding online security and tech support reduces churn probability.


## 🧠 What I Learned
- Practical application of binary classification to real-world business problems.
- Importance of feature engineering and domain context in churn modeling.
- Model interpretability using SHAP to explain individual predictions.
- How businesses can save revenue by acting on predicted risk levels.

---
