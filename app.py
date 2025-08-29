from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import BaseModel, ValidationError
import joblib
import numpy as np
import logging

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "super-secret"
jwt = JWTManager(app)

# Rate Limiting (5 requests/minute per IP)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["5 per minute"])

# Load model
model = joblib.load("fraud_model.pkl")

# Logging setup
logging.basicConfig(filename="fraud_detection.log", level=logging.INFO)

# Input schema
class Transaction(BaseModel):
    amount: float
    device: str
    location: str
    hour: int

# Dummy login
@app.route("/login", methods=["POST"])
def login():
    creds = request.get_json()
    if creds["username"] == "admin" and creds["password"] == "password":
        token = create_access_token(identity=creds["username"])
        return jsonify(access_token=token), 200
    return jsonify({"msg": "Invalid credentials"}), 401

# Predict
@app.route("/predict", methods=["POST"])
@jwt_required()
@limiter.limit("5 per minute")
def predict():
    try:
        data = request.get_json()
        txn = Transaction(**data)
        input_vector = np.array([[txn.amount, txn.device == "mobile", txn.location == "Delhi", txn.hour]])
        prob = model.predict_proba(input_vector)[0][1]
        prediction = int(prob > 0.5)

        # Log
        logging.info(f"Prediction: {prediction}, Confidence: {prob:.4f}")

        return jsonify({
            "prediction": prediction,
            "confidence": round(prob, 4)
        })
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
