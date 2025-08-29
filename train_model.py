import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, fbeta_score, precision_recall_curve

# Load dataset
df = pd.read_csv("synthetic_fraud_data.csv")
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(xgb, param_grid, scoring='recall', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Predict
xgb_preds = best_model.predict(X_test)
xgb_probs = best_model.predict_proba(X_test)[:, 1]

# Metrics
print("Best Params:", grid.best_params_)
print(classification_report(y_test, xgb_preds))
print("F2 Score:", fbeta_score(y_test, xgb_preds, beta=2))

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, xgb_probs)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("precision_recall_curve.png")
plt.close()

# Feature Importance
plot_importance(best_model)
plt.title("XGBoost Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# SHAP
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], show=False)
plt.savefig("shap_summary_plot.png")
plt.close()

# Save model
joblib.dump(best_model, "fraud_model.pkl")
