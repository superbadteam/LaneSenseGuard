
import cv2
import urllib.request
import numpy as np

# cam2 = "http://169.254.142.134:8000/stream.mjpg"
cam2 = "http://192.168.137.9:8080/?action=stream"
stream = urllib.request.urlopen(cam2)
bytes = bytes()
while True:
    bytes += stream.read(1024)
    a = bytes.find(b'\xff\xd8')
    b = bytes.find(b'\xff\xd9')
    if a != -1 and b != -1:
        jpg = bytes[a:b+2]
        bytes = bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow('i', i)
        if cv2.waitKey(1) == 27:
            exit(0)