## Xây dựng một ứng dụng nhận diện đối tượng theo thời gian thực
import cv2
from ultralytics import YOLO
import time

model = YOLO('./model_1_dataset_1/weights/best.pt')  # Đường dẫn đến mô hình đã được huấn luyện

class_names = ['gloves', 'hands', 'head', 'helmet', 'no-gloves', 'no-helmet', 'no-safety-vest', 'person', 'safety-suit', 'safety-vest']

cap = cv2.VideoCapture('test_dataset/busy-roof-construction-SBV-300154131-preview.mp4')
cap.set(3, 640)
cap.set(4, 640)

new_width = 800
new_height = 600

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (new_width, new_height))

    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # bounding box coordinates
            x1, y1, x2, y2, = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # class name
            cls = int(box.cls[0])
            print("Class name: ", class_names[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0 , 0)
            thickness = 2

            cv2.putText(frame, class_names[cls], org, font, fontScale, color, thickness, cv2.LINE_AA)

        # resized_frame = cv2.resize(frame, (new_width, new_height))

        # cv2.imshow("Webcam", resized_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()