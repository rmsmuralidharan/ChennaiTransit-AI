# 🚇 Chennai Transit AI

## Passenger Demand Prediction System

Chennai Transit AI is a machine learning-powered passenger demand prediction system designed to estimate the expected number of passengers at transit stations based on station, location, time, weather, and recent passenger demand.

The system provides predictions through a FastAPI backend and an interactive Streamlit dashboard.

---

## 🚀 Features

- Passenger demand prediction
- Interactive Streamlit dashboard
- FastAPI prediction API
- XGBoost machine learning model
- Station selection
- Date and time-based prediction
- Weather-based demand factors
- Peak-hour detection
- Weekend detection
- Recent demand analysis
- Prediction history

---

## 🧠 Machine Learning Models

The following models were evaluated:

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Ridge Regression | 15.4845 | 28.5460 | 0.9287 |
| Transformer | 7.0189 | 11.3636 | 0.9887 |
| XGBoost | 4.1933 | 7.0182 | 0.9957 |

### 🏆 Final Model

**XGBoost** was selected as the final production model because it achieved the best performance on the test dataset.

---

## 📊 Features Used

The model uses features such as:

- Station ID
- Zone
- Latitude
- Longitude
- Month
- Day
- Hour
- Minutes
- Day of week
- Weekend indicator
- Morning peak indicator
- Evening peak indicator
- Peak-hour indicator
- Temperature
- Humidity
- Rainfall
- Weather condition
- Demand profile
- Previous passenger count
- Rolling average passenger demand

---

## 🏗️ System Architecture

```text
User
  │
  ▼
Streamlit Dashboard
  │
  ▼
FastAPI Backend
  │
  ▼
Prediction Pipeline
  │
  ├── Preprocessor
  │
  └── XGBoost Model
  │
  ▼
Passenger Demand Prediction