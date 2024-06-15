import os
import shutil
from PIL import Image

# Đường dẫn đến thư mục chứa dữ liệu ảnh
data_dir = '/Users/phuc1403/Downloads/mrlEyes_2018_01'

# Tạo các thư mục đích (nếu chưa tồn tại)
folder_open = os.path.join(data_dir, 'folder_0')
folder_close = os.path.join(data_dir, 'folder_1')

os.makedirs(folder_open, exist_ok=True)
os.makedirs(folder_close, exist_ok=True)

# Hàm để resize ảnh về kích thước 32x32
def resize_image(filepath):
    try:
        img = Image.open(filepath)
        img = img.resize((32, 32), Image.ANTIALIAS)
        img.save(filepath)
    except Exception as e:
        print(f"Error resizing image {filepath}: {e}")

# Hàm đệ quy để duyệt qua các thư mục con và xử lý tệp ảnh
def process_images(directory, max_images_per_category=10000):
    count_open = 0
    count_closed = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                filepath = os.path.join(root, filename)
                try:
                    # Tách lấy trạng thái mắt và thông tin kính từ tên file
                    parts = filename.split('_')
                    eye_state = int(parts[4])  # Vị trí 5 chứa thông tin về trạng thái mắt
                    glasses_status = int(parts[3])  # Vị trí 4 chứa thông tin về kính

                    # Chỉ xử lý ảnh không có kính
                    if glasses_status == 0:
                        # Resize ảnh về kích thước 32x32 trước khi di chuyển
                        resize_image(filepath)

                        # Điều hướng tới thư mục đích tùy theo trạng thái mắt
                        if eye_state == 0 and count_closed < max_images_per_category:
                            destination_folder = folder_close
                            count_closed += 1
                        elif eye_state == 1 and count_open < max_images_per_category:
                            destination_folder = folder_open
                            count_open += 1
                        else:
                            continue

                        # Đường dẫn thư mục đích và di chuyển tệp
                        destination_path = os.path.join(destination_folder, filename)
                        shutil.move(filepath, destination_path)
                        
                        # Kiểm tra nếu đã đạt đủ số lượng ảnh cho cả hai trạng thái
                        if count_open >= max_images_per_category and count_closed >= max_images_per_category:
                            print(f"Đã di chuyển đủ 10.000 ảnh cho mỗi trạng thái mắt.")
                            return
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
    
    print(f"Đã di chuyển tổng cộng {count_open} ảnh mắt mở và {count_closed} ảnh mắt nhắm.")

# Bắt đầu xử lý từ thư mục gốc
process_images(data_dir)

print("Đã phân loại xong và resize các ảnh.")