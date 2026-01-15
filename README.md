Bank Customer Churn Prediction : 

An end-to-end Machine Learning project that predicts whether a bank customer is likely to churn, along with an interpretable churn risk score, deployed using an interactive Streamlit dashboard.

Project Objective : 

To help banks identify high-risk customers early and enable proactive retention strategies by predicting customer churn based on behavioral and financial data.

Machine Learning Approach : 

Performed data cleaning and preprocessing
Removed non-informative identifier columns
Applied one-hot encoding for categorical features
Engineered meaningful features such as:
Balance–Salary Ratio
High-Value Customer Indicator
Trained a Random Forest Classifier with:
Stratified train–test split
Tuned hyperparameters
Class imbalance handling
Evaluated using precision, recall, and F1-score (focus on churn recall)

Churn Risk Prediction : 

Outputs probability-based churn risk instead of hard labels

Streamlit Dashboard

Interactive UI to input customer details
Displays churn probability and risk category
Ensures consistency between training and inference by aligning feature space
Designed for real-world decision support, not just model output

Key Challenges Solved

Feature mismatch between training and deployment
Model bias due to class imbalance
Overconfident predictions from hard thresholds
Interpretability of ML predictions
