from flask import Flask, request, jsonify
from flask_cors import CORS
from duDoan import predict_image
from PIL import Image
import io
import os

# ==== Config ====
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)

# ==== Helper ====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==== Routes ====
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            image = Image.open(io.BytesIO(file.read()))
            predictions = predict_image(image, topk=5)
            result = {label: f"{prob}%" for label, prob in predictions}

            return jsonify({
                'prediction': result
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

# ==== Run ====
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
