import os
import re

# Đường dẫn thư mục chứa ảnh và nhãn
image_dir = './SH17_Original-1/train/images'
label_dir = './SH17_Original-1/train/labels'

# Hàm xóa phần đuôi '_jpeg.rf.xxx...' khỏi tên file
def rename_files_in_dir(directory):
    for filename in os.listdir(directory):
        # Tách phần mở rộng
        name, ext = os.path.splitext(filename)
        
        # Kiểm tra có chuỗi '_jpeg.rf.' không
        match = re.match(r'^(.+?)_jpeg\.rf\..+$', name)
        if match:
            new_name = match.group(1) + ext
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            # Tránh ghi đè nếu file mới đã tồn tại
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_name}")
            else:
                print(f"Skipped (exists): {new_name}")

# Thực hiện đổi tên cho thư mục ảnh và nhãn
rename_files_in_dir(image_dir)
rename_files_in_dir(label_dir)