from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained XGBoost model at startup
MODEL_PATH = "ml_models/xgb_best_with_null.joblib"
MODEL_PATH_CALIBRATED = "ml_models/xgb_calibrated_with_nulls.joblib"

model = joblib.load(MODEL_PATH)

model_calibrated = joblib.load(MODEL_PATH_CALIBRATED)

@app.route("/", methods=["GET"])
def health_check():
    """
    Simple health check to verify the app is running.
    """
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON payload:
    {
      "features": [f1, f2, ..., fn]
    }
    Returns:
    {
      "probability": p
    }
    """
    data = request.get_json(force=True)
    features = data.get("features")
    
    
    if features is None:
        return jsonify({"error": "Missing 'features' in request body."}), 400
    
    features = [np.nan if f is None or f == "null" else f for f in features]
    
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list of numbers."}), 400

    try:
        # Convert input list to a 2D numpy array for a single sample
        x = np.array(features, dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid feature vector: {e}"}), 400

    # Compute probability of the positive class (class 1)
    try:
        proba = model.predict_proba(x)[0, 1]
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    return jsonify({"probability": float(proba)}), 200

@app.route("/predict-calibrated", methods=["POST"])
def predict():
    """
    Expects a JSON payload:
    {
      "features": [f1, f2, ..., fn]
    }
    Returns:
    {
      "probability": p
    }
    """
    data = request.get_json(force=True)
    features = data.get("features")
    
    
    if features is None:
        return jsonify({"error": "Missing 'features' in request body."}), 400
    
    features = [np.nan if f is None or f == "null" else f for f in features]
    
    if not isinstance(features, list):
        return jsonify({"error": "'features' must be a list of numbers."}), 400

    try:
        # Convert input list to a 2D numpy array for a single sample
        x = np.array(features, dtype=float).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid feature vector: {e}"}), 400

    # Compute probability of the positive class (class 1)
    try:
        proba = model_calibrated.predict_proba(x)[0, 1]
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    return jsonify({"probability": float(proba)}), 200

if __name__ == "__main__":
    # When running locally: python app.py
    app.run(host="0.0.0.0", port=5000, debug=True)