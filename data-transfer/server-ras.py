import asyncio
import websockets

connected_clients = set()

async def handler(websocket, path):
    global system_status
    count_distracted = 0
    count_wrong_lane = 0

    connected_clients.add(websocket)
    print('Got connection from', websocket.remote_address)
    try:
         while True:
         # Receive a message from the client
            message = await websocket.recv()
            print('Received:', message)
            for client in connected_clients:
                await client.send(message)
            data = message.split(':')

            # handle lane
            if data[0] == 'lane':
                if data[1] == 'wrong':
                    count_distracted += 1
                    if count_distracted == 3:
                        print('Driver is distracted')
                    if count_distracted == 5:
                        print('Driver is extremely distracted')
                else:
                    count_distracted = 0
            print('Lane:', data[1], count_distracted)

            # handle face
            if data[0] == 'face':
                print('Face:', data[1])
                if data[1] == 'detected':
                    count_wrong_lane += 1
                    if count_wrong_lane == 3:
                        print('Driver is not looking at the road')
                    if count_wrong_lane == 5:
                        print('Driver is extremely not looking at the road')
                else:
                    count_wrong_lane = 0

    except websockets.ConnectionClosed:
        print('Client disconnected')
    except Exception as e:
        print(e)
    finally:
        connected_clients.remove(websocket)

async def run_server():
    server = await websockets.serve(handler, "0.0.0.0", 12345)
    print('Server listening on ws://0.0.0.0:12345')
    await server.wait_closed()

### Client WebSocket

ON = 1
OFF = 0
system_status = ON

async def run_client():
    global system_status
    uri = "ws://103.77.246.238:8765"  # Thay thế bằng IP của VPS

    async with websockets.connect(uri) as websocket:
        try:
            while True:
                data = await websocket.recv()
                data = data.split(':')

                if data[0].strip() == "system":
                    if data[1] == 'on':
                        system_status = ON
                        print('System is ON')
                    else:
                        system_status = OFF
                        print('System is OFF')
                else:
                    print('Not a system message', data[0].strip(),"system")
                print('Received from server:', data[0], data[1])
        except websockets.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(e)

async def main():
    server_task = asyncio.create_task(run_server())
    client_task = asyncio.create_task(run_client())

    await asyncio.gather(server_task, client_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
