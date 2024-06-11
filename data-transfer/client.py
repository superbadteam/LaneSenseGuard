import socket

s = socket.socket()
s.connect(('169.254.142.134', 12345))

while True:
    data = input('Enter data to send: ')
    s.send(data.encode())
    print('Received:', s.recv(1024).decode())
    if data == 'exit':
        break
s.close()
