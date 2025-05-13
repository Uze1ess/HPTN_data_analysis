import cv2
from ultralytics import YOLO

model = YOLO('./model_2/weights/best.pt')
cap = cv2.VideoCapture('test_dataset/busy-roof-construction-SBV-300154131-preview.mp4')
# cap = cv2.VideoCapture('test_dataset/2341409-hd_2048_960_30fps.mp4')
# cap = cv2.VideoCapture('test_dataset/busy-roof-construction-SBV-300154131-preview.mp4')

max_width = 1200

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    height, width = annotated_frame.shape[:2]
    if width > max_width:
        scale = max_width / width
        annotated_frame = cv2.resize(annotated_frame, (int(width * scale), int(height * scale)))

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()