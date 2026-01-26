import cv2

# Try with DirectShow backend (good for Windows)
index = 0  # Change to 1 or 2 to test others
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

print(f"Opened (index {index}):", cap.isOpened())

if not cap.isOpened():
    print(f"❌ Cannot open camera at index {index}")
    exit()

while True:
    ret, frame = cap.read()
    print("ret:", ret)
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
