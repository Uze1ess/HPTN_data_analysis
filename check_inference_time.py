import cv2
import time
from ultralytics import YOLO

# Load YOLO model (tùy theo phiên bản, ví dụ: YOLOv5s hoặc YOLOv8n)
model = YOLO("./model_1_dataset_1/weights/best.pt")  # Hoặc "yolov8n.pt"

# Mở video từ file hoặc camera
video_path = "test_dataset/busy-roof-construction-SBV-300154131-preview.mp4"  # Hoặc 0 nếu dùng webcam
cap = cv2.VideoCapture(video_path)

frame_skip = 5  # Số khung hình bỏ qua
frame_count = 0

# Thống kê hiệu suất
frame_count = 0
# frame_count_processed = 0
total_processing_time = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.time()

    # Chạy object detection
    results = model(frame)
    # frame_count_processed += 1

    frame_end = time.time()
    processing_time = frame_end - frame_start
    total_processing_time += processing_time
    frame_count += 1

    # Tính FPS và thời gian xử lý
    current_fps = 1.0 / processing_time if processing_time > 0 else 0
    avg_fps = frame_count / (frame_end - start_time)
    avg_service_time = total_processing_time / frame_count

    # Hiển thị thông tin lên frame
    text1 = f"Current FPS: {current_fps:.2f}"
    text2 = f"Avg FPS: {avg_fps:.2f}"
    text3 = f"Service Time: {processing_time*1000:.2f} ms"
    cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(frame, text3, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Hiển thị khung hình
    cv2.imshow("YOLO Performance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Thống kê cuối cùng
print(f"Tổng số khung: {frame_count}")
print(f"FPS trung bình: {avg_fps:.2f}")
print(f"Thời gian xử lý trung bình mỗi khung: {avg_service_time*1000:.2f} ms")
# print(f"Số khung hình đã xử lý: {frame_count_processed}")