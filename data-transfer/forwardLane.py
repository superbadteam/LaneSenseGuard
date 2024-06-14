import asyncio
import websockets
import cv2
import numpy as np
import socket
import threading

clients = set()

async def handler(websocket, path):
    clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        clients.discard(websocket)  # Sử dụng discard thay vì remove để tránh KeyError

def start_websocket_server():
    return websockets.serve(handler, "0.0.0.0", 5001)

def start_tcp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('0.0.0.0', 5002))
    sock.listen(1)
    conn, addr = sock.accept()
    print('Connected by', addr)

    bytes_data = b''
    while True:
        try:
            data = conn.recv(4096)
            if not data:
                break
            bytes_data += data
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    frame_data = buffer.tobytes()
                    # Sử dụng run_coroutine_threadsafe để gọi send_frame_to_clients từ vòng lặp asyncio
                    asyncio.run_coroutine_threadsafe(send_frame_to_clients(frame_data), loop)
        except Exception as e:
            print(f"Error: {e}")
            pass

async def send_frame_to_clients(frame_data):
    for client in clients.copy():
        try:
            await client.send(frame_data)
        except:
            clients.discard(client)  # Sử dụng discard thay vì remove để tránh KeyError

if __name__ == '__main__':
    tcp_thread = threading.Thread(target=start_tcp_server)
    tcp_thread.daemon = True
    tcp_thread.start()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_websocket_server())
    loop.run_forever()
