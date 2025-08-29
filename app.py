from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView
from pydantic import BaseModel, ValidationError
import joblib
import numpy as np
import datetime

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "super-secret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fraud_detection.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

jwt = JWTManager(app)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["5 per minute"])
db = SQLAlchemy(app)
admin = Admin(app, name="Fraud Monitor", template_mode="bootstrap3")

# Load model
model = joblib.load("fraud_model.pkl")

# Input Schema
class Transaction(BaseModel):
    amount: float
    device: str
    location: str
    hour: int

# DB Model
class FraudLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    amount = db.Column(db.Float)
    device = db.Column(db.String(50))
    location = db.Column(db.String(50))
    hour = db.Column(db.Integer)
    prediction = db.Column(db.Integer)
    confidence = db.Column(db.Float)

admin.add_view(ModelView(FraudLog, db.session))

# Routes
@app.route("/login", methods=["POST"])
def login():
    creds = request.get_json()
    if creds["username"] == "admin" and creds["password"] == "password":
        token = create_access_token(identity=creds["username"])
        return jsonify(access_token=token), 200
    return jsonify({"msg": "Invalid credentials"}), 401

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

        # Log to DB
        log = FraudLog(
            amount=txn.amount,
            device=txn.device,
            location=txn.location,
            hour=txn.hour,
            prediction=prediction,
            confidence=prob
        )
        db.session.add(log)
        db.session.commit()

        return jsonify({
            "prediction": prediction,
            "confidence": round(prob, 4)
        })
    except ValidationError as e:
        return jsonify({"error": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
