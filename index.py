from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np

app = Flask(__name__)

def detect_emotion_simple(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # PROVEN WEB_CAM parameters (from GeeksforGeeks working example)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,    # Standard for webcams
        minNeighbors=5,     # Standard 
        minSize=(30, 30)    # Smallest face size
    )
    
    print(f"Detected {len(faces)} faces")  # Debug line
    
    if len(faces) > 0:
        return f"happy (face size: {faces[0][2]})"
    return "no_face"


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
        
        emotion = detect_emotion_simple(img)
        return jsonify({'emotion': emotion})
    except:
        return jsonify({'error': 'Detection failed'}), 500

if __name__ == '__main__':
    app.run()
