from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import onnxruntime as ort
import os

app = Flask(__name__)

# Emotion labels — FERPlus ONNX model output order
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

# Load face detector (built into OpenCV, no download needed)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load ONNX emotion model once at startup
MODEL_PATH = https://huggingface.co/aasthachaudhary25/emotion-detector/resolve/main/emotion-ferplus-2.onnx
emotion_session = ort.InferenceSession(MODEL_PATH)
input_name = emotion_session.get_inputs()[0].name


def preprocess_face(face_gray):
    """Resize and normalize face ROI for FERPlus ONNX model (64x64 grayscale)."""
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
        gray = cv2.equalizeHist(gray)  # normalize lighting

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,       # lowered for better detection
            minSize=(50, 50),     # works for farther/smaller faces
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return jsonify({'emotion': 'no_face'})

        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]

        # ONNX inference
        face_input = preprocess_face(face_roi)
        outputs = emotion_session.run(None, {input_name: face_input})
        scores = outputs[0][0]  # shape: (8,)

        # Softmax for probabilities
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
