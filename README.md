# 📈 Retail Sales Forecasting Dashboard

> End-to-end ML forecasting pipeline on Kaggle Superstore data — **ARIMA vs Prophet** with interactive 1–12 month forecast horizon selector.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![Prophet](https://img.shields.io/badge/Prophet-Meta-blue)](https://facebook.github.io/prophet/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue)](https://plotly.com)

---

## 🔍 What it does

An end-to-end retail sales forecasting system that trains two models, compares them on held-out test data, and renders an interactive forecast dashboard.

| Feature | Detail |
|---|---|
| Dataset | Kaggle Superstore (4 years, 9,994 orders) |
| Models | ARIMA(2,1,2) vs Facebook Prophet |
| Evaluation | MAE on 15% held-out test set |
| Forecasting | 4–52 week horizon with confidence intervals |
| Categories | Furniture · Office Supplies · Technology |

---

## 📊 Key Finding

> **Prophet outperforms ARIMA by ~12% MAE** — significantly better at handling the November–December seasonal spikes driven by holiday retail demand.

---

## 🚀 Run Locally

```bash
git clone https://github.com/archit-dhar/retail-sales-forecasting
cd retail-sales-forecasting
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏗️ Pipeline Architecture

```
Raw Superstore Data (CSV / simulated)
        │
        ▼
  Data Cleaning & Feature Engineering
  (weekly aggregation, date parsing)
        │
        ├─────────────────────────────┐
        ▼                             ▼
  ARIMA(2,1,2)               Facebook Prophet
  (statsmodels)              (yearly seasonality,
        │                   changepoint detection)
        │                             │
        └──────────┬──────────────────┘
                   ▼
         MAE Comparison on Test Set
                   │
                   ▼
       Streamlit Interactive Dashboard
       (forecast plot + confidence bands)
```

---

## 📁 Project Structure

```
retail-sales-forecasting/
├── app.py              # Main Streamlit app
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

`Python` · `Facebook Prophet` · `ARIMA (statsmodels)` · `Pandas` · `NumPy` · `Plotly` · `Streamlit` · `scikit-learn`

---

## 👤 Author

**Archit Dhar** · [LinkedIn](https://linkedin.com/in/archit-dhar) · B.E. AI & Data Science, VESIT Mumbai 2025
