import socket
import RPi.GPIO as GPIO

 

GPIO.setmode(GPIO.BCM)

GPIO.setup(17, GPIO.OUT)

 

# host is my ip address
# get your ip address and add to host
host = '0.0.0.0'
port = 12345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

# print my ip address
print('Server IP:', socket.gethostbyname(socket.gethostname()))
s.listen(5)
print('Server listening....')

c, addr = s.accept()
print('Got connection from', addr)

while True:
    data = c.recv(1024)
    if not data:
        break
    data = data.decode()
    if data == 'on':
        GPIO.output(17, True)
    elif data == 'off':
        GPIO.output(17, False)
    print('Received:', data.decode())
    c.send(data)
c.close()
