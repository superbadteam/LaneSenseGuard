import socket
import json
import threading

# Thông tin server
server_host = '0.0.0.0'
server_port = 12345

# Thông tin client
client_host = 'your_vps_ip'  # Thay thế bằng IP của VPS
client_port = 12345

def run_server():
    # Khởi tạo socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((server_host, server_port))

    # Lắng nghe kết nối
    print('Server IP:', socket.gethostbyname(socket.gethostname()))
    server_socket.listen(5)
    print('Server listening....')

    while True:
        client_socket, addr = server_socket.accept()
        print('Got connection from', addr)
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    print('Client disconnected')
                    break
                data = data.decode()
                # split string like lane:right
                data = data.split(':')
                print('Received:', data)
                if data[0] == 'lane':
                    print('Lane:', data[1])
                if data[0] == 'face':
                    print('Face:', data[1])
                # Gửi phản hồi về client
                response = f"Received {data[0]} data with value {data[1]}"
                client_socket.send(response.encode())
            except Exception as e:
                print(e)
                break
        client_socket.close()

def run_client():
    # Khởi tạo socket client
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((client_host, client_port))

    # Gửi dữ liệu đến server
    messages = [
        "lane:right",
        "face:detected"
    ]

    for message in messages:
        client_socket.send(message.encode())
        print('Sent:', message)
        
        # Nhận phản hồi từ server
        data = client_socket.recv(1024)
        if not data:
            break
        print('Received from server:', data.decode())
    
    # Liên tục lắng nghe dữ liệu từ server
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print('Received from server:', data.decode())
    except Exception as e:
        print(e)
    
    # Đóng kết nối
    client_socket.close()

# Tạo và chạy thread cho server
server_thread = threading.Thread(target=run_server)
server_thread.start()

# Tạo và chạy thread cho client
client_thread = threading.Thread(target=run_client)
client_thread.start()

# Đợi các thread hoàn thành
server_thread.join()
client_thread.join()
