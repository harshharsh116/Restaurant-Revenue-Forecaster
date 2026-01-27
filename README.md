# Restaurant-Revenue-Forecaster
No fancy time series — just solid regression ML to predict monthly earnings from location, restaurant type, city demographics, open days, and hidden features. Fast, interpretable, and ready to deploy.Predict next month's revenue today with basic machine learning. Regression-based approach using restaurant profile data.

A lightweight, non-time-series machine learning tool to predict monthly restaurant revenue.
Uses regression models (Linear Regression, Random Forest, XGBoost, etc.) trained on static restaurant features:

City and city group (big cities vs. others)
Restaurant type (food court, inline, etc.)
Obfuscated demographic, real-estate, and commercial variables
Goal: Provide fast, accurate monthly revenue estimates to support decisions on expansion, budgeting, or performance evaluation — without needing historical daily/weekly sales sequences.

Key highlights:

Simple & interpretable models
Feature engineering (log revenue, encoding, scaling)
Model evaluation with RMSE/MAE
Python stack: pandas, scikit-learn, xgboost, matplotlib/seaborn
