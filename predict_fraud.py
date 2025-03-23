import requests
import json
import sys
import argparse

def predict_fraud(transaction_data, host='localhost', port=5000):
    """Send transaction data to API and get fraud prediction"""
    url = f'http://{host}:{port}/predict'
    
    try:
        response = requests.post(url, json=transaction_data)
        
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Result:")
            print(json.dumps(result, indent=4))
            
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

def main():
    parser = argparse.ArgumentParser(description='Predict fraud for a transaction using the Fraud Detection API')
    parser.add_argument('--host', default='localhost', help='API host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, help='API port (default: 5000)')
    parser.add_argument('--file', help='JSON file containing transaction data')
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Error reading file {args.file}:", e)
            return
    else:
        print("Enter your JSON transaction data (press Ctrl+Z and Enter when done):")
        print("\nExample format:")
        print('''{
    "transaction_id": "ANON_123",
    "transaction_date": "2024-12-13 11:17:53",
    "transaction_amount": 199.0,
    "transaction_channel": "mobile",
    "transaction_payment_mode": 10,
    "payment_gateway_bank": 0,
    "payer_email": "email_hash",
    "payer_mobile": "XXXXX967.0",
    "payer_browser": 517,
    "payee_id": "ANON_119",
    "payee_ip": "ip_hash"
}''')
        print("\nNote: transaction_payment_mode should be a number")
        print("\nPaste your JSON here:")
        
        try:
            # Read all input until EOF (Ctrl+Z)
            json_str = sys.stdin.read()
            
            if not json_str.strip():
                print("No input provided")
                return
                
            json_data = json.loads(json_str)
            
        except json.JSONDecodeError as e:
            print("\nError: Invalid JSON format -", str(e))
            return
        except Exception as e:
            print("\nError:", str(e))
            return

    # Ensure transaction_payment_mode is numeric
    if 'transaction_payment_mode' in json_data:
        try:
            json_data['transaction_payment_mode'] = float(json_data['transaction_payment_mode'])
        except (ValueError, TypeError):
            print("Error: transaction_payment_mode must be a number")
            return
    
    # Add _anonymous suffix to required fields
    fields_to_anonymize = [
        'transaction_id', 'payee_id', 'transaction_payment_mode',
        'payment_gateway_bank', 'payer_email', 'payer_mobile',
        'payer_browser', 'payee_ip'
    ]
    
    for field in fields_to_anonymize:
        if field in json_data:
            json_data[f"{field}_anonymous"] = json_data.pop(field)
    
    predict_fraud(json_data, args.host, args.port)

if __name__ == "__main__":
    main()
