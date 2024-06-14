import asyncio
import websockets

connected_clients = set()

async def handler(websocket, path):
    # Thêm client vào danh sách các client kết nối
    connected_clients.add(websocket)
    # print chào mừng khi có client kết nối
    print(f"Client connected: {websocket}")
    try:
        async for message in websocket:
            print(f"Received message: {message}")
            # Chuyển dữ liệu tới tất cả các client đang kết nối
            await asyncio.wait([client.send(message) for client in connected_clients])
    finally:
        # Loại bỏ client khỏi danh sách khi ngắt kết nối
        connected_clients.remove(websocket)

start_server = websockets.serve(handler, "0.0.0.0", 8765)

print("Server started at ws://localhost:8765")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
