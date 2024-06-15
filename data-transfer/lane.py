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
            # Chuyển đổi frame ảnh từ client sang dạng np.array
            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Gửi frame tới tất cả các client khác
            await send_frame_to_clients(frame, websocket)
    except websockets.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")
    finally:
        clients.discard(websocket)  # Sử dụng discard thay vì remove để tránh KeyError

async def send_frame_to_clients(frame, sender):
    # Nén khung hình thành dạng jpg để gửi đi
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    frame_data = buffer.tobytes()
    
    for client in clients.copy():
        if client != sender:  # Không gửi lại cho client đã gửi
            try:
                await client.send(frame_data)
            except websockets.ConnectionClosed:
                print(f"Removing client {client.remote_address}")
                clients.discard(client)  # Sử dụng discard thay vì remove để tránh KeyError
            except Exception as e:
                print(f"Error sending to client: {e}")
                clients.discard(client)  # Sử dụng discard thay vì remove để tránh KeyError

async def main():
    websocket_server = await websockets.serve(handler, "0.0.0.0", 5001)
    print('WebSocket Server listening on ws://0.0.0.0:5001')
    await websocket_server.wait_closed()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
