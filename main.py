from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_image
from PIL import Image
import io
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
CORS(app)  # bật CORS cho toàn bộ app

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            # Đọc ảnh trực tiếp từ bộ nhớ
            image = Image.open(io.BytesIO(file.read()))
            prediction = predict_image(image)  # truyền ảnh PIL vào hàm predict
            return jsonify({'prediction': prediction}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

