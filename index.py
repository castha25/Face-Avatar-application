from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from face_emotion_app import detect_emotion  # Adjust import as needed
import io

app = Flask(__name__)

@app.route('/')
def home():
    return open('index.html').read()  # Serve your HTML

@app.route('/api/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Use your existing emotion detection
        emotion = detect_emotion(img)  # From face_emotion_app.py
        
        return jsonify({'emotion': emotion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
