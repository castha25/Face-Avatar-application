from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from fer import FER  # Your original library

app = Flask(__name__)

# Init FER detector (like your face_emotion_app.py)
emotion_detector = FER()

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
        
        # EXACTLY your face_emotion_app.py logic
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emotion, score = emotion_detector.top_emotion(rgb_frame)
        
        if emotion:
            result = f"{emotion}: {int(score * 100)}%"
        else:
            result = "no_face"
            
        return jsonify({'emotion': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
