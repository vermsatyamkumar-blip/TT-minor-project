from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from model import train_model

app = Flask(__name__)

# ---------- Train model & compute metrics once at startup ----------
trained_model, mse, rmse, mae, r2, X_train, X_test, y_train, y_test, predictions, df = train_model()

METRICS = {
    "r2": round(r2, 4),
    "mse": round(mse, 4),
    "rmse": round(rmse, 4),
    "mae": round(mae, 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "n_total": len(df),
    "coefficients": {col: round(c, 4) for col, c in zip(X_train.columns, trained_model.coef_)},
    "intercept": round(trained_model.intercept_, 4),
    # Actual vs predicted for chart (first 50 test points)
    "actual": y_test.values[:50].tolist(),
    "predicted": predictions[:50].tolist(),
}


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        study_hours = float(data["study_hours_per_week"])
        attendance = float(data["attendance_pct"])
        previous_marks = float(data["previous_marks"])

        features = np.array([[study_hours, attendance, previous_marks]])
        prediction = trained_model.predict(features)[0]
        prediction = round(max(0, min(100, prediction)), 2)  # clamp 0-100

        return jsonify({"success": True, "prediction": prediction})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/metrics")
def metrics():
    return jsonify(METRICS)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
