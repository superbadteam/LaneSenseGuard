import asyncio
import websockets
import json
import time
import RPi.GPIO as GPIO

# GPIO setup
BUZZER_PIN = 4
BUTTON_LEFT = 2
BUTTON_RIGHT = 3
LED_LEFT = 17
LED_RIGHT = 27
LED_POWER = 22
POWER_SWITCH = 10


GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_LEFT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON_RIGHT, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(LED_LEFT, GPIO.OUT)
GPIO.setup(LED_RIGHT, GPIO.OUT)
GPIO.setup(LED_POWER, GPIO.OUT)
GPIO.setup(POWER_SWITCH, GPIO.IN, pull_up_down=GPIO.PUD_UP)

connected_clients = set()

count_distracted = 0
count_wrong_lane = 0

ON = 1
OFF = 0

system_status = ON
led_right = OFF
led_left = OFF
# Kiểm tra trạng thái của các nút
def check_button_status():
    global led_left, led_right, system_status
    if GPIO.input(BUTTON_LEFT) == GPIO.LOW:
        led_left = ON
    else:
        led_left = OFF

    if GPIO.input(BUTTON_RIGHT) == GPIO.LOW:
        led_right = ON
    else:
        led_right = OFF

    if GPIO.input(POWER_SWITCH) == GPIO.LOW:
    #    switch_status and convert to int
        system_status = 1 - system_status
        GPIO.output(BUZZER_PIN, 1)
    else:
        GPIO.output(BUZZER_PIN, 0)

    GPIO.output(LED_POWER, system_status)

    

def handle_system():
    if system_status == ON:
        GPIO.output(LED_LEFT, led_left)

        GPIO.output(LED_RIGHT, led_right)

        if count_distracted > 3:
            GPIO.output(BUZZER_PIN, 1)
        else:
            GPIO.output(BUZZER_PIN, 0)
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
        check_button_status()
        handle_system()
        status_message = json.dumps({
            "system_status": system_status,
            "led_right": led_right,
            "led_left": led_left,
            "time":  time.time(),
            "buzzer": GPIO.input(BUZZER_PIN)
            })
        await websocket.send(status_message)
        await asyncio.sleep(0.5)  # Gửi system_status mỗi 5 giây

async def run_client():
    global system_status, led_right, led_left
    uri = "ws://103.77.246.238:8765"  # Thay thế bằng IP của VPS

    async with websockets.connect(uri) as websocket:
        send_task = asyncio.create_task(send_system_status(websocket))
        try:
           async for message in websocket:
                try:
                    data = json.loads(message)
                    print('Received:', data)

                    system_status_receive = data.get('system_status', system_status)
                    led_right_receive = data.get('led_right', led_right)
                    led_left_receive = data.get('led_left', led_left)


                    # check null or empty
                    if system_status_receive is not None:
                        system_status = system_status_receive

                    if led_right_receive is not None:
                        led_right = led_right_receive

                    if led_left_receive is not None:
                        led_left = led_left_receive

                except Exception as e:
                    print(e)
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
    init_values()

    server_task = asyncio.create_task(run_server())
    client_task = asyncio.create_task(run_client())

    await asyncio.gather(server_task, client_task)


def init_values():
    GPIO.output(BUZZER_PIN, 0)
    GPIO.output(LED_LEFT, 0)
    GPIO.output(LED_RIGHT, 0)
    GPIO.output(LED_POWER, 0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user. Shutting down...")

    finally:
        GPIO.cleanup()

