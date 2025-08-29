from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
import joblib
import pandas as pd
import numpy as np
import logging
import time

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'your-secure-secret-key'  # Change this in production!

jwt = JWTManager(app)

# Load trained model
model = joblib.load("fraud_model.pkl")

# Set up logging
logging.basicConfig(filename='fraud_detection.log', level=logging.INFO, format='%(asctime)s %(message)s')

@app.route('/login', methods=['POST'])
def login():
    username = request.json.get("username", None)
    password = request.json.get("password", None)
    if username != "admin" or password != "password":
        return jsonify({"msg": "Bad credentials"}), 401
    access_token = create_access_token(identity=username)
    return jsonify(access_token=access_token), 200

@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        start_time = time.time()
        proba = model.predict_proba(df)[0][1]
        prediction = int(proba > 0.3)
        latency = round((time.time() - start_time) * 1000, 2)  # ms

        log_data = {
            "input": data,
            "prediction": prediction,
            "probability": proba,
            "latency_ms": latency
        }
        logging.info(log_data)

        return jsonify({
            "prediction": prediction,
            "fraud_probability": round(proba, 4),
            "latency_ms": latency
        })
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Invalid input or server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)