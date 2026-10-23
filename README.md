# 🔍 AI Fraud Detection System

A machine learning web app that predicts whether a transaction is **fraudulent or legitimate** and shows the **fraud probability**.

**Live app:** https://ratnadip-fraud-detection.streamlit.app

## How it works

1. Enter four transaction details in the web app.
2. A trained Random Forest model checks the transaction.
3. The app shows the result (legitimate or fraudulent) and the fraud probability in percent.

## Input features

| Feature | Meaning |
|---|---|
| `amount` | Value of the transaction |
| `transaction_hour` | Hour of the day (0 to 23) |
| `distance_from_home` | Distance in km from the customer's usual location |
| `transactions_last_24h` | Number of transactions in the previous 24 hours |

## Tech stack

- Python
- pandas
- scikit-learn (Random Forest, 100 trees)
- joblib
- Streamlit

## Model results

The data is split 75% for training and 25% for testing.

| Metric | Score |
|---|---|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-score | 100% |

> **Note:** the test set contains only 5 transactions (3 legitimate, 2 fraud). These scores show that the pipeline works, but they do not prove real-world performance. A larger dataset is needed for a reliable estimate.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `fraud_model.pkl` | Trained model |
| `requirements.txt` | Python packages needed to run the app |
| `train_model.py` | Trains and evaluates the model (run locally) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model, place `transactions.csv` in the same folder and run:

```bash
python train_model.py
```

## Future scope

- Train on a larger, more realistic dataset
- Add more features such as merchant type, device and country
- Compare several algorithms
- Upload a CSV to check many transactions at once

## Author

**Ratnadip Kale**, KBPM
