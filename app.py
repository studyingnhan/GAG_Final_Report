import os
import base64
import time
import uuid
from io import BytesIO
from typing import List, Tuple, Optional

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename
from keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.applications import resnet50, inception_v3, efficientnet

# ─────────────────────────────── NEW: fast detector ──────────────────────────────
from utils.mp_detector import detect_faces_mp  # MediaPipe BlazeFace

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

UPLOAD_FOLDER = 'static/uploaded'
OUTPUT_FOLDER = 'static/output'

AVAILABLE_MODELS = {
    'ResNet50'      : 'ResNet50_finetuned.keras',
    'ResNet101'     : 'ResNet101_finetuned.keras',  # remove if file không tồn tại
    'InceptionV3'   : 'InceptionV3_finetuned.keras',
    'EfficientNetB0': 'EfficientNetB0_finetuned.keras',
}

MODEL_PREPROC = {
    'ResNet50'      : resnet50.preprocess_input,
    'ResNet101'     : resnet50.preprocess_input,
    'InceptionV3'   : inception_v3.preprocess_input,
    'EfficientNetB0': efficientnet.preprocess_input,
}

DEFAULT_MODEL      = 'ResNet50'
DEFAULT_DETECTOR   = 'mp'   # 'mp' (MediaPipe) | 'mtcnn'
MIN_FACE_SIZE      = 80     # pixels (not used in MP)
MARGIN_RATIO       = 0.20   # 20 % around bbox

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    OUTPUT_FOLDER=OUTPUT_FOLDER,
    MAX_CONTENT_LENGTH=5 * 1024 * 1024  # 5 MB
)

# --------------------------------------------------
# Globals
# --------------------------------------------------
gender_labels = ['Female', 'Male']
age_labels    = ['Child', 'Teen', 'Adult', 'Elderly']

detector_mtcnn = MTCNN()  # vẫn giữ MTCNN cho tuỳ chọn accurate
model:   Optional["keras.Model"] = None
current_model_name: str = DEFAULT_MODEL

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def load_selected_model(model_name: str) -> bool:
    """Load Keras model lazily; tránh load lại nếu đã ở bộ nhớ."""
    global model, current_model_name

    if model_name == current_model_name and model is not None:
        return True

    model_path = os.path.join(MODEL_DIR, AVAILABLE_MODELS.get(model_name, ''))
    if not os.path.exists(model_path):
        return False

    model = load_model(model_path, compile=False)
    current_model_name = model_name
    return True


def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """Resize 224×224 & chuẩn hoá theo backbone."""
    face_resized = cv2.resize(face_bgr, (224, 224))
    pre_fn = MODEL_PREPROC.get(current_model_name)
    if pre_fn is None:
        x = face_resized.astype(np.float32) / 255.0
    else:
        x = pre_fn(face_resized.astype(np.float32))
    return np.expand_dims(x, 0)


def _annotate(image: np.ndarray, box: Tuple[int, int, int, int], label: str) -> None:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def predict_frame(frame_bgr: np.ndarray, detector_name: str = DEFAULT_DETECTOR):
    """Return tuple(base64_jpeg | None, labels, fps)."""
    if model is None:
        raise RuntimeError('Model not loaded.')

    t0 = time.time()

    # ---------- detect faces ----------
    if detector_name == 'mp':
        boxes = detect_faces_mp(frame_bgr)
    else:  # 'mtcnn'
        faces = detector_mtcnn.detect_faces(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes = [f['box'] for f in faces]

    if not boxes:
        return None, [], 0.0

    # ---------- preprocess & predict ----------
    prepared = [preprocess_face(frame_bgr[y:y+h, x:x+w])
                for (x, y, w, h) in boxes]
    batch = np.vstack(prepared)
    gender_preds, age_preds = model.predict(batch, verbose=0)

    # ---------- annotate ----------
    labels = []
    for (box, g_pred, a_pred) in zip(boxes, gender_preds, age_preds):
        label = f"{gender_labels[int(np.argmax(g_pred))]}, {age_labels[int(np.argmax(a_pred))]}"
        _annotate(frame_bgr, box, label)
        labels.append(label)

    # encode ảnh → base64 JPEG (quality 80)
    _, buf = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    img_b64 = base64.b64encode(buf).decode('utf-8')
    fps = 1.0 / (time.time() - t0 + 1e-6)
    return img_b64, labels, round(fps, 1)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html', models=list(AVAILABLE_MODELS.keys()), current=DEFAULT_MODEL)


@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model', current_model_name)
    # ➕ Lấy detector từ form (mp hoặc mtcnn)
    detector_name = request.form.get('detector', DEFAULT_DETECTOR)  # 'mp' hoặc 'mtcnn'    
    if not load_selected_model(model_name):
        return jsonify({'error': 'Model not found'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)
    frame = cv2.imread(upload_path)
    
    if frame is None:
        return jsonify({'error': 'Cannot read image'}), 400
    img_b64, labels, _ = predict_frame(frame, detector_name)  # dùng detector được chọn
    
    if img_b64 is None:
        return jsonify({'error': 'No face detected'}), 200
    return jsonify({'data': img_b64, 'predictions': labels})


@app.route('/webcam', methods=['POST'])
def webcam():
    data = request.json or {}
    img_b64 = data.get('img')
    model_name = data.get('model', current_model_name)
    detector_name = data.get('detector', DEFAULT_DETECTOR)

    if not load_selected_model(model_name):
        return jsonify({'error': 'Model not found'}), 404

    try:
        img_bytes = base64.b64decode(img_b64.split(',')[-1])
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({'error': 'Invalid image data'}), 400

    img_b64_out, labels, fps = predict_frame(frame, detector_name)
    if img_b64_out is None:
        return jsonify({'error': 'No face'}), 200

    return jsonify({'data': img_b64_out, 'predictions': labels, 'fps': fps})


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    load_selected_model(DEFAULT_MODEL)
    app.run(host='0.0.0.0', port=5000, debug=True)
