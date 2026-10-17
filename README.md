Monthly Price Forecasting System

A machine-learning project that forecasts future monthly prices using historical data, with a FastAPI backend and a Django dashboard.

Project Description

The Monthly Price Forecasting System demonstrates the application of machine learning, data processing, REST API development, and web deployment. It uses historical price data to generate future monthly forecasts and makes the results accessible through a web dashboard and API.

Live Application

Dashboard: https://monthly-price-forecast-dashboard.onrender.com
API Documentation: https://price-forecast-api-3xtd.onrender.com/docs
Forecast Endpoint: https://price-forecast-api-3xtd.onrender.com/forecast?months=6

Key Features

Historical price data processing

Monthly price forecasting

Baseline model implementation

Model evaluation and comparison

REST API built with FastAPI

Django-based dashboard

Cloud deployment using Render

Technologies

Python · Pandas · Scikit-learn · Joblib · FastAPI · Django · REST API · Git · GitHub · Render

Machine Learning Concepts

Time-Series Forecasting

Time-series forecasting uses historical observations collected over time to estimate future values. It is commonly used in sales, demand, inventory, and price analysis.

Naive Forecasting

The current deployed model uses Naive Forecasting, a baseline method that predicts the next period's value using the latest observed value.

Forecast = Latest observed value

For example, if the latest observed price is ₹15,750, the model predicts ₹15,750 for the next month. Consequently, forecasts remain constant unless the latest observed value changes.

Model Evaluation

Forecasting models can be assessed using metrics such as:

MAE: Mean Absolute Error

MSE: Mean Squared Error

RMSE: Root Mean Squared Error

MAPE: Mean Absolute Percentage Error, when appropriate

Chronological validation is important when evaluating time-series forecasting models.

System Architecture

Historical Data → Data Processing → Forecasting Model → FastAPI → Django Dashboard → Forecast Results

The FastAPI service generates forecast results. The Django dashboard communicates with the API and presents the results to users.

Project Objectives

Apply machine-learning concepts to historical price data.

Develop a baseline forecasting solution.

Understand model evaluation and comparison.

Build and integrate REST API services.

Develop a web-based dashboard.

Deploy a Python application to the cloud.

Current Limitations

The deployed baseline model repeats the latest observed price.

It does not independently capture price trends or seasonality.

Forecast quality depends on historical data and model selection.

Forecasts are estimates, not guarantees of future prices.

Future Enhancements

Compare forecasting algorithms such as Linear Regression and time-series models.

Improve chronological model validation.

Add forecast-versus-actual visualizations.

Include prediction intervals where appropriate.

Improve testing, input validation, and dashboard functionality.

Learning Outcomes

This project provided practical experience with:

Python-based data analysis

Machine-learning experimentation

Forecast evaluation concepts

FastAPI REST API development

Django web application development

API integration

GitHub version control

Cloud deployment

Author
