# Fraud Detection API

This project implements a machine learning model for detecting fraudulent transactions using Random Forest Classifier. The model is exposed through a REST API that can be deployed to Render.

## Features

- Random Forest Classifier with optimized parameters
- RESTful API endpoints for fraud detection
- Feature engineering for transaction data
- Multiple encoding methods for categorical variables
- Threshold optimization for fraud detection

## Project Structure

- `improved_fraud_detection.py`: Model training script
- `api.py`: Flask API implementation
- `wsgi.py`: WSGI entry point
- `requirements.txt`: Package dependencies
- `Procfile`: Render deployment configuration

## Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python improved_fraud_detection.py
```

3. Run the API locally:
```bash
python api.py
```

## API Endpoints

### Health Check
- URL: `/health`
- Method: `GET`
- Response: Status of model loading

### Predict Fraud
- URL: `/predict`
- Method: `POST`
- Request Body:
```json
{
    "transaction_id": "TRANS123",
    "transaction_amount": 1000.0,
    "transaction_date": "2023-01-01T12:00:00",
    "transaction_channel": "online"
}
```
- Response:
```json
{
    "transaction_id": "TRANS123",
    "fraud_probability": 0.15,
    "is_fraud": 0,
    "threshold": 0.5
}
```

## Deployment to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
   - Environment Variables: None required
4. Deploy the service

## Model Performance

- ROC-AUC Score: 0.8913
- Precision for fraud detection: 0.12
- Recall for fraud detection: 0.86
- F1-score for fraud detection: 0.20

## Data Distribution

- Training data: 0.0064% fraud cases
- Predictions: 0.0634% fraud cases (94 fraudulent transactions out of 148,228 total)
