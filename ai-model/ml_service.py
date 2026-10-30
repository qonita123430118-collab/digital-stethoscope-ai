# ml_service.py — jalankan: python ml_service.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler satu kali saat startup
print("Loading model...")
model  = joblib.load('heart_model.pkl')
scaler = joblib.load('scaler.pkl')
print("Model ready!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data   = request.json
        signal = data['signal']

        if len(signal) != 187:
            signal = np.interp(
                np.linspace(0, len(signal) - 1, 187),
                np.arange(len(signal)),
                signal
            ).tolist()

        features = np.array(signal, dtype=np.float64).reshape(1, -1)
        
        # TAMBAHKAN: Normalisasi sinyal ESP32 ke range yang mirip PTBDB
        # sebelum dimasukkan ke scaler yang sudah fit dengan data PTBDB
        sig_min = features.min()
        sig_max = features.max()
        if sig_max - sig_min > 0:
            # Min-max normalize ke range -1 sampai 1 (mirip ECG)
            features = 2 * (features - sig_min) / (sig_max - sig_min) - 1
        
        features_scaled = scaler.transform(features)
        
        prediction  = model.predict(features_scaled)[0]
        proba       = model.predict_proba(features_scaled)[0]
        confidence  = float(proba.max()) * 100

        result = 'NORMAL' if prediction == 1 else 'ABNORMAL'

        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)