import socket

s = socket.socket()
s.connect(('103.77.246.238', 12345))

while True:
    data = input('Enter data to send: ')
    s.send(data.encode())
    recieve = s.recv(1024)
    print('Received:', recieve.decode())
    if data == 'exit':
        break
s.close()
