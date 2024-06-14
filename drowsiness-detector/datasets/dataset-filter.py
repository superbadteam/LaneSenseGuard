import os
import shutil
from PIL import Image

# Đường dẫn đến thư mục chứa dữ liệu ảnh
data_dir = '/Users/phuc1403/Downloads/untitled folder/Train'

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
def process_images(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            # Nếu là thư mục, tiếp tục đệ quy vào thư mục con
            process_images(filepath)
        elif filename.endswith('.png'):  # Chỉ xét các file ảnh PNG, bạn có thể điều chỉnh nếu có định dạng khác
            # Resize ảnh về kích thước 32x32 trước khi di chuyển
            resize_image(filepath)
            
            # Tách lấy trạng thái mắt từ tên file
            parts = filename.split('_')
            eye_state = int(parts[4])  # Vị trí 5 chứa thông tin về trạng thái mắt

            # Điều hướng tới thư mục đích tùy theo trạng thái mắt
            if eye_state == 0:
                destination_folder = folder_close
            elif eye_state == 1:
                destination_folder = folder_open
            
            # Đường dẫn thư mục đích và di chuyển tệp
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(filepath, destination_path)

# Bắt đầu xử lý từ thư mục gốc
process_images(data_dir)

print("Đã phân loại xong và resize các ảnh.")
