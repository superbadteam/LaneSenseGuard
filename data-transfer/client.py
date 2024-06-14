import asyncio
import websockets

async def send_and_receive():
    # Thay đổi "your_ip_address" thành địa chỉ IP cụ thể của server
    uri = "ws://10.10.29.119:12345"
    async with websockets.connect(uri) as websocket:
        while True:
            # Get user input
            message = input("Enter message: ")

            # Send the message to the server
            await websocket.send(message)

            # Receive a message from the server
            response = await websocket.recv()

            # Print the received message
            print("Received:", response)

asyncio.get_event_loop().run_until_complete(send_and_receive())
