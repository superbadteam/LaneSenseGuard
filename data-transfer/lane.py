import asyncio
import websockets
import cv2
import numpy as np

clients = set()

async def handler(websocket, path):
    clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            # Xử lý các thông điệp từ client nếu cần
            pass
    except websockets.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    finally:
        clients.discard(websocket)  # Sử dụng discard thay vì remove để tránh KeyError

async def send_frame_to_clients(frame_data):
    for client in clients.copy():
        try:
            await client.send(frame_data)
        except websockets.ConnectionClosed:
            print(f"Removing client {client.remote_address}")
            clients.discard(client)  # Sử dụng discard thay vì remove để tránh KeyError
        except Exception as e:
            print(f"Error sending to client: {e}")
            clients.discard(client)  # Sử dụng discard thay vì remove để tránh KeyError

async def process_frames():
    bytes_data = b''
    stream = cv2.VideoCapture(0)  # Sử dụng webcam cho stream
    while True:
        ret, frame = stream.read()
        if not ret:
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_data = buffer.tobytes()
        await send_frame_to_clients(frame_data)
        await asyncio.sleep(0.1)  # Điều chỉnh thời gian ngủ tùy theo yêu cầu của bạn

async def main():
    websocket_server = await websockets.serve(handler, "0.0.0.0", 5001)
    print('WebSocket Server listening on ws://0.0.0.0:5001')
    await asyncio.gather(websocket_server.wait_closed(), process_frames())

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
