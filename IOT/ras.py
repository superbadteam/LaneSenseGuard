# import RPi.GPIO as GPIO
# from time import sleep

# try: 
#     pass
#     GPIO.setmode(GPIO.BCM)
#     GPIO.setup(17, GPIO.OUT)
#     GPIO.setup(2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#     while True:
#         if GPIO.input(2) == GPIO.LOW:
#             # GPIO.output(17, True)
#             print('Button pressed')
#             sleep(0.5)
# except KeyboardInterrupt:
#     pass
#     # GPIO.cleanup()
# except:
#     pass
# finally:
#     GPIO.cleanup()


import RPi.GPIO as GPIO
from time import sleep
try: 
    pass
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    GPIO.setup(2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(4, GPIO.OUT)
    while True:
        if GPIO.input(2) == False:
            GPIO.output(4, GPIO.HIGH)
            sleep(1)
            print('Button pressed')
        else:
            GPIO.output(4, GPIO.LOW)
except KeyboardInterrupt:
    pass
    # GPIO.cleanup()
except Exception as e:
    print(e)
    pass
finally:
    GPIO.cleanup()


