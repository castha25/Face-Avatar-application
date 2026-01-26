import cv2
import os
import numpy as np
from fer import FER
import mediapipe as mp
from moviepy.editor import VideoFileClip

# Setup
output_path = "output.avi"
final_output = "final_output.mp4"
frame_width, frame_height = 640, 480
fps = 20

# FER setup
emotion_detector = FER(mtcnn=True)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# OpenCV video capture
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Recording... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    emotion, score = emotion_detector.top_emotion(frame)
    emotion_text = f"{emotion} ({score:.2f})" if emotion else "No emotion"

    # Detect face landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for point in landmarks.landmark:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Overlay emotion
    cv2.putText(frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show & record
    cv2.imshow("Live Expression Capture", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording finished, converting to mp4...")

# Convert .avi to .mp4 using MoviePy
clip = VideoFileClip(output_path)
clip.write_videofile(final_output, codec='libx264')

# Cleanup .avi if needed
os.remove(output_path)

print(f"✅ Done! Saved as: {final_output}")
