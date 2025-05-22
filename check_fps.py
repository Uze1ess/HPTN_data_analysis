import cv2
import time

cap = cv2.VideoCapture('test_dataset/busy-roof-construction-SBV-300154131-preview.mp4')

def check_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def check_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count

def check_fps_from_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

if __name__ == "__main__":
    video_path = 'test_dataset/busy-roof-construction-SBV-300154131-preview.mp4'
    camera_index = 0  # Thay đổi nếu cần

    # Kiểm tra FPS từ video
    fps_video = check_fps(video_path)
    print(f"FPS từ video: {fps_video}")

    # Kiểm tra số lượng frame từ video
    frame_count = check_frame_count(video_path)
    print(f"Số lượng frame trong video: {frame_count}")

    # Kiểm tra FPS từ camera
    fps_camera = check_fps_from_camera(camera_index)
    print(f"FPS từ camera: {fps_camera}")