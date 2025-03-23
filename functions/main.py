import functions_framework
from flask import request, jsonify

@functions_framework.http
def fraud_detection(request):
    request_json = request.get_json()
    
    if not request_json or "transaction" not in request_json:
        return jsonify({"error": "Invalid request"}), 400

    transaction = request_json["transaction"]

    # Example fraud detection logic
    if transaction.get("amount", 0) > 5000:
        return jsonify({"fraud": True, "message": "High transaction amount detected"})
    
    return jsonify({"fraud": False, "message": "Transaction looks safe"})
