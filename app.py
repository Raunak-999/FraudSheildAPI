from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)

# Load the saved model and preprocessing objects
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoding_maps = joblib.load('encoders.pkl')
    amount_bounds = joblib.load('amount_bounds.pkl')
    optimal_threshold = joblib.load('optimal_threshold.pkl')
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    print("Please run improved_fraud_detection.py first to generate the model files.")
    raise

@app.route('/')
def home():
    return '''
    <h1>Fraud Detection API</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><b>GET /health</b>: Check API health</li>
        <li><b>POST /predict</b>: Predict fraud for a transaction</li>
    </ul>
    <h2>Example POST /predict request:</h2>
    <pre>
    {
        "transaction_id_anonymous": "12345",
        "transaction_date": "2023-03-23T10:30:00",
        "transaction_amount": 1000.0,
        "transaction_channel": "web",
        "payee_id_anonymous": "P123",
        "transaction_payment_mode_anonymous": "credit_card",
        "payment_gateway_bank_anonymous": "bank1",
        "payer_browser_anonymous": "chrome",
        "payer_email_anonymous": "user@example.com",
        "payee_ip_anonymous": "192.168.1.1",
        "payer_mobile_anonymous": "1234567890"
    }
    </pre>
    '''

def preprocess_single_transaction(transaction_data):
    """Preprocess a single transaction using the same steps as the training script."""
    try:
        # Create a copy of the data and remove transaction_id
        data = transaction_data.copy()
        transaction_id = data.pop('transaction_id_anonymous', None)
        
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([data])
        
        # Handle missing values
        df['transaction_amount'] = pd.to_numeric(df['transaction_amount'], errors='coerce').fillna(0)
        
        # Convert date and create time features
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_hour'] = df['transaction_date'].dt.hour
        df['transaction_day'] = df['transaction_date'].dt.day
        df['transaction_dow'] = df['transaction_date'].dt.dayofweek
        df['transaction_weekend'] = df['transaction_dow'].isin([5, 6]).astype(int)
        
        # Create cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['transaction_day']/31)
        df['day_cos'] = np.cos(2 * np.pi * df['transaction_day']/31)
        
        # Handle transaction amount outliers
        df['transaction_amount'] = df['transaction_amount'].clip(
            lower=amount_bounds['lower'],
            upper=amount_bounds['upper']
        )
        
        # Create amount-based features
        df['amount_log'] = np.log1p(df['transaction_amount'])
        
        # Convert numeric fields
        numeric_fields = [
            'transaction_payment_mode_anonymous',
            'payment_gateway_bank_anonymous',
            'payer_browser_anonymous'
        ]
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
        
        # Categorical encoding
        categorical_cols = ['transaction_channel', 'payee_id_anonymous']
        
        for col in categorical_cols:
            # Convert to string for consistent encoding
            df[col] = df[col].astype(str)
            # Risk encoding
            df[f'{col}_risk'] = df[col].map(encoding_maps.get(f'{col}_risk', {})).fillna(0)
            # Frequency encoding
            df[f'{col}_freq'] = df[col].map(encoding_maps.get(f'{col}_freq', {})).fillna(0)
            # Label encoding
            df[f'{col}_label'] = df[col].map(encoding_maps.get(f'{col}_labels', {})).fillna(-1)
        
        # Drop unnecessary columns
        columns_to_drop = categorical_cols + ['transaction_date']
        df = df.drop(columns_to_drop, axis=1)
        
        # Ensure all columns are numeric
        for col in df.columns:
            if not np.issubdtype(df[col].dtype, np.number):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Get the expected feature names from the model
        expected_features = model.feature_names_in_
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # Add missing features with default value
        
        # Reorder columns to match model's expected features
        df = df[expected_features]
        
        # Scale numerical features
        numerical_cols = ['transaction_amount', 'amount_log'] + \
                        [col for col in df.columns if '_risk' in col or '_freq' in col]
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        return df, transaction_id
    except Exception as e:
        raise ValueError(f"Error preprocessing transaction: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Preprocess the transaction
        processed_data, transaction_id = preprocess_single_transaction(data)
        
        if not transaction_id:
            return jsonify({'error': 'transaction_id_anonymous is required'}), 400
        
        # Make prediction
        fraud_probability = model.predict_proba(processed_data)[0][1]
        is_fraud = int(fraud_probability >= optimal_threshold)
        
        # Return prediction
        return jsonify({
            'transaction_id_anonymous': transaction_id,
            'is_fraud': is_fraud,
            'fraud_probability': float(fraud_probability)
        })
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
