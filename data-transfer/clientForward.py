import cv2
import urllib.request
import numpy as np
import socket

# URL của camera stream
cam2 = "http://192.168.145.37:8080/?action=stream"
stream = urllib.request.urlopen(cam2)
bytes_data = bytes()

# Địa chỉ IP công khai của server
server_ip = '103.77.246.238'
server_port = 5002

# Set up TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((server_ip, server_port))

while True:
    bytes_data += stream.read(1024)
    a = bytes_data.find(b'\xff\xd8')
    b = bytes_data.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if frame is not None:
            # Resize the frame to reduce the size
            frame = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            sock.sendall(buffer.tobytes())
