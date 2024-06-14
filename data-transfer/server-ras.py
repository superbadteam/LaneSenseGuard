import asyncio
import websockets
import json
import time

connected_clients = set()

count_distracted = 0
count_wrong_lane = 0

ON = 1
OFF = 0

system_status = ON
led_right = OFF
led_left = OFF

# handle data from client
def handle_data_from_AI_server(data):
    global count_distracted, count_wrong_lane
    # handle lane
    if data[0] == 'lane':
        print('Lane:', data[1], count_distracted)
        if data[1] == 'wrong':
            count_distracted += 1
            if count_distracted == 3:
                print('Driver is distracted')
            if count_distracted == 5:
                print('Driver is extremely distracted')
        else:
            count_distracted = 0

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

async def handler(websocket, path):
    global system_status

    connected_clients.add(websocket)
    print('Got connection from', websocket.remote_address)
    try:
        async for message in websocket:
            for client in connected_clients:
                await client.send(message)
            data = message.split(':')
            print('Received:', data)

            try:
                handle_data_from_AI_server(data)
            except Exception as e:
                pass

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


async def send_system_status(websocket):
    global system_status
    while True:
        status_message = json.dumps({
            "system_status": system_status,
            "led_right": led_right,
            "led_left": led_left,
            "time":  time.time()
            })
        await websocket.send(status_message)
        print(f"Sent system_status: {system_status}")
        await asyncio.sleep(0.5)  # Gửi system_status mỗi 5 giây

async def run_client():
    global system_status
    uri = "ws://103.77.246.238:8765"  # Thay thế bằng IP của VPS

    async with websockets.connect(uri) as websocket:
        send_task = asyncio.create_task(send_system_status(websocket))
        try:
           async for message in websocket:
                try:
                    data = json.loads(message)
                    print('Received:', data)
                except json.JSONDecodeError:
                    print('Received:', message)
                    pass
                # data = data.split(':')

                # if data[0].strip() == "system":
                #     if data[1] == 'on':
                #         system_status = ON
                #         print('System is ON')
                #     else:
                #         system_status = OFF
                #         print('System is OFF')
                # else:
                #     print('Not a system message', data[0].strip(),"system")
                # print('Received from server:', data[0], data[1])
        except websockets.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(e)
        finally:
            send_task.cancel()

async def main():
    server_task = asyncio.create_task(run_server())
    client_task = asyncio.create_task(run_client())

    await asyncio.gather(server_task, client_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")
