from flask import Flask, request, jsonify
import joblib
import numpy as np
from pymongo import MongoClient

model = joblib.load("fraud_model.pkl")

client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]
collection = db["transactions"]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    fields = ['transaction_amount', 'location_score', 'device_score', 'transaction_time',
              'customer_age', 'transaction_frequency']

    if not all(k in data for k in fields):
        return jsonify({"error": "Missing fields"}), 400

    values = np.array([[data[k] for k in fields]])
    prob = model.predict_proba(values)[0][1]
    prediction = int(prob > 0.5)

    result = {
        "fraud_probability": round(prob, 4),
        "is_fraud": prediction
    }

    data.update(result)
    collection.insert_one(data)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
