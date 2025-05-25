from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_image
from PIL import Image
import io
import os
import time

# ==== Config ====
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SAVE_DIR = 'saved_images'
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# ==== Helper ====
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_save_paths(filename):
    # Tạo tên file với timestamp tránh trùng
    base_name = os.path.splitext(filename)[0]
    timestamp = int(time.time() * 1000)
    original_path = os.path.join(SAVE_DIR, f"{base_name}_{timestamp}_original.jpg")
    resized_path = os.path.join(SAVE_DIR, f"{base_name}_{timestamp}_resized.jpg")
    preprocessed_path = os.path.join(SAVE_DIR, f"{base_name}_{timestamp}_preprocessed.jpg")
    return original_path, resized_path, preprocessed_path

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
            orig_path, resized_path, preproc_path = get_save_paths(file.filename)

            # Gọi predict_image với lưu ảnh
            predictions = predict_image(image, topk=5,
                                        save_original_path=orig_path,
                                        save_resized_path=resized_path,
                                        save_preprocessed_tensor_path=preproc_path)
            result = {label: f"{prob}%" for label, prob in predictions}

            # Trả thêm đường dẫn ảnh đã lưu (nếu bạn cần frontend xem)
            return jsonify({
                'prediction': result,
                'saved_images': {
                    'original': orig_path,
                    'resized': resized_path,
                    'preprocessed': preproc_path
                }
            }), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file format'}), 400

# ==== Run ====
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
