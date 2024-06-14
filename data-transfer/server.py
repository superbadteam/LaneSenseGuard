import socket
import json
# Thông tin server
host = '0.0.0.0'
port = 12345

# Khởi tạo socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

# Lắng nghe kết nối
print('Server IP:', socket.gethostbyname(socket.gethostname()))
s.listen(5)
print('Server listening....')

while True:
    c, addr = s.accept()
    print('Got connection from', addr)
    while True:
        try:
            data = c.recv(1024)
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
                
            # c.send(data.encode())
        except Exception as e:
            print(e)
            break
    c.close()
