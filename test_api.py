import requests
import json

# Your specific transaction data
transaction_data = {
    "transaction_id_anonymous": "ANON_287602",  # Mapping to match API expectations
    "transaction_date": "2024-12-13 11:17:53",
    "transaction_amount": 199.0,
    "transaction_channel": "mobile",
    "transaction_payment_mode_anonymous": 10,  # Mapping to match API expectations
    "payment_gateway_bank_anonymous": 0,  # Mapping to match API expectations
    "payer_email_anonymous": "01e539d52a86183530375b04b281abda87d90f30d35c981235bb3220cb21cf51",
    "payer_mobile_anonymous": "XXXXX967.0",  # Mapping to match API expectations
    "payer_browser_anonymous": 517,  # Mapping to match API expectations
    "payee_id_anonymous": "ANON_119",  # Mapping to match API expectations
    "payee_ip_anonymous": "b298ae3f549799b26d6f95df25f28388438fe22598f3bff59fbdb489ca9d0ecf"  # Mapping to match API expectations
}

# API endpoint
url = 'http://localhost:5000/predict'

try:
    # First check if the API is healthy
    health_response = requests.get('http://localhost:5000/health')
    print("\nHealth check status:", health_response.json())

    print("\nSending the following transaction data to API:")
    print(json.dumps(transaction_data, indent=4))
    
    # Make the prediction request
    response = requests.post(url, json=transaction_data)
    
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(json.dumps(result, indent=4))
        
        # Add a more user-friendly interpretation
        if result['is_fraud'] == 1:
            print("\n⚠️ WARNING: This transaction is flagged as potentially fraudulent!")
            print(f"Fraud probability: {result['fraud_probability']*100:.2f}%")
        else:
            print("\n✅ This transaction appears to be legitimate.")
            print(f"Fraud probability: {result['fraud_probability']*100:.2f}%")
    else:
        print("\nError:", response.status_code)
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Error connecting to the API:", e)
except Exception as e:
    print("An unexpected error occurred:", e)
