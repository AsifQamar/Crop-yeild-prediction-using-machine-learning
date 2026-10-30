"""
Crop Yield Prediction - Flask Backend
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

app = Flask(__name__)

# Load artifacts
model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
model_name = joblib.load(os.path.join(MODEL_DIR, 'best_model_name.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))

with open(os.path.join(MODEL_DIR, 'dropdown_data.json'), 'r') as f:
    dropdown_data = json.load(f)

with open(os.path.join(MODEL_DIR, 'model_comparison.json'), 'r') as f:
    comparison = json.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/dropdown-data')
def get_dropdown_data():
    """Return the crop, season, state lists for the form dropdowns."""
    return jsonify(dropdown_data)


@app.route('/api/model-info')
def model_info():
    """Return the model comparison metrics."""
    return jsonify(comparison)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Encode categorical features
        crop_encoded = label_encoders['Crop'].transform([data['crop']])[0]
        season_encoded = label_encoders['Season'].transform([data['season']])[0]
        state_encoded = label_encoders['State'].transform([data['state']])[0]

        # Build feature array in the same order as training
        features = np.array([[
            crop_encoded,
            int(data['crop_year']),
            season_encoded,
            state_encoded,
            float(data['area']),
            float(data['annual_rainfall']),
            float(data['fertilizer']),
            float(data['pesticide']),
            float(data['avg_temperature']),
            float(data['max_temperature']),
            float(data['min_temperature']),
        ]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'success': True,
            'predicted_yield': round(float(prediction), 4),
            'model_used': model_name
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/eda-plots')
def eda_plots():
    """Return list of EDA plot filenames."""
    eda_dir = os.path.join(BASE_DIR, 'eda_plots')
    plots = [f for f in os.listdir(eda_dir) if f.endswith('.png')]
    return jsonify(plots)


# Serve EDA plot images
@app.route('/eda_plots/<filename>')
def serve_plot(filename):
    from flask import send_from_directory
    return send_from_directory(os.path.join(BASE_DIR, 'eda_plots'), filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

