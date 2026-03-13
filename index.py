from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import onnxruntime as ort
import os
import urllib.request

app = Flask(__name__)

EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

MODEL_URL = "https://huggingface.co/aasthachaudhary25/emotion-detector/resolve/main/emotion-ferplus-2.onnx"
MODEL_PATH = "/tmp/emotion-ferplus-2.onnx"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading emotion model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")
    return ort.InferenceSession(MODEL_PATH)

emotion_session = load_model()
input_name = emotion_session.get_inputs()[0].name

def preprocess_face(face_gray):
    face_resized = cv2.resize(face_gray, (64, 64))
    face_input = face_resized.astype(np.float32).reshape(1, 1, 64, 64)
    return face_input

@app.route('/')
def home():
    return open('index.html').read()

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'emotion': 'no_face'})

        img = cv2.resize(img, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return jsonify({'emotion': 'no_face'})

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]

        face_input = preprocess_face(face_roi)
        outputs = emotion_session.run(None, {input_name: face_input})
        scores = outputs[0][0]

        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        top_idx = int(np.argmax(probs))
        emotion = EMOTIONS[top_idx]
        confidence = int(probs[top_idx] * 100)

        return jsonify({'emotion': f'{emotion}: {confidence}%'})

    except Exception as e:
        return jsonify({'error': str(e), 'emotion': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
