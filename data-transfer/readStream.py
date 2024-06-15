
import cv2
import urllib.request
import numpy as np
import tensorflow
from tensorflow import keras
import tarfile
import shutil
import json
import math
import os

def Draw_Lines(image, lines, color=[255, 0, 0], thickness=12):
  for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def Hough_Transformation(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    Draw_Lines(line_image, lines, thickness=2)

    return line_image, lines

def Get_Lane_Area(image, vertices):
  mask = np.zeros_like(image)

  if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
  else:
        ignore_mask_color = 255

  cv2.fillPoly(mask, vertices, ignore_mask_color)
  masked_image = cv2.bitwise_and(image, mask)

  return masked_image, mask

def make_line_points(y1, y2, line):

    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return [x1, y1, x2, y2]# ((x1, y1), (x2, y2))

def connect_lane_lines(lines, imshape):
    try:
        left_lines    = [] # (slope, intercept)
        left_weights  = [] # (length,)
        right_lines   = [] # (slope, intercept)
        right_weights = [] # (length,)
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        
        # add more weight to longer lines    
        left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
        right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
        
        y1 = 390 # bottom of the image
        y2 = 295        # slightly lower than the middle

        left_line  = make_line_points(y1, y2, left_lane)
        right_line = make_line_points(y1, y2, right_lane)
        
        connected_lines = np.int32([[left_line, right_line]])
        
        return connected_lines # left_line, right_line
    except Exception as e:
        return None
        # print(e)

def Detect_Lane(image, plot_image=False,
                kernel_size=5, canny_low_threshold=50, canny_high_threshold=150,
                hough_rho=1, hough_theta=np.pi/180, hough_threshold=20,
                hough_min_line_len=20, hough_max_line_gap=300):
       
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray_blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Canny Edge Detection
    edges = cv2.Canny(gray_blur, canny_low_threshold, canny_high_threshold)

    # Concentrate the location of edge detection
    image_shape = image.shape
    
    # vertices = np.array([[(400,400),(680, 290), (850, 290), (1150,400)]], dtype=np.int32) # (590, 1640, 3)
    vertices = np.array([[(125, 390),(280, 295), (350, 295), (515,390)]], dtype=np.int32) # (480, 640, 3)
    
    masked_edges, mask = Get_Lane_Area(edges, vertices=vertices)

    # Detect lines using Hough transform on an edge detected image
    lines_image, lines = Hough_Transformation(masked_edges,
                                  rho=hough_rho, theta=hough_theta, threshold=hough_threshold,
                                  min_line_len=hough_min_line_len, max_line_gap=hough_max_line_gap)

    # Merge 'original' image with 'lines' image
    # result = Merge_Image(lines_image, image)
    connected_lines = connect_lane_lines(lines, image_shape)
    
    # Plot the images
    if plot_image:
        plt.figure(figsize=[16, 9])
        for i, img in enumerate(['gray', 'gray_blur', 'edges', 'mask', 'masked_edges', 'lines_image', 'result']):
            Plot_Image(eval(img), img, (4,2, i+1))
            plt.axis('off')
            plt.show()
    return connected_lines # result, connected_lines, gray, gray_blur, edges, masked_edges, lines_image

def Plot_Image(image, title, subplot_pos):
    plt.subplot(*subplot_pos)
    plt.title(title)
    if len(image.shape) == 3:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap='gray')

def Merge_Image(line_image, initial_image):
  # Check if images are None
  if line_image is None or initial_image is None:
      raise ValueError("One or both input images are None.")

  # Check if images have the same dimensions
  if line_image.shape != initial_image.shape:
      print(line_image.shape)
      print(initial_image.shape)

  # Check if images have the same data type
  if line_image.dtype != initial_image.dtype:
      raise ValueError("Input images must have the same data type.")
  return cv2.addWeighted(initial_image, 1.0, line_image, 1.0, 0.0)

def draw_connected_lane_lines(img, connected_lines):
    # Get a copy of the original image
    lines_img = np.copy(img)*0
    Draw_Lines(lines_img, connected_lines, thickness=12)
    
    return Merge_Image(lines_img, img)

def process_image(image):
    
    # Find the lanes
    connected_lines = Detect_Lane(image, plot_image=False,
                                    kernel_size=5, canny_low_threshold=50, canny_high_threshold=150,
                                    hough_rho=1, hough_theta=np.pi/180, hough_threshold=20,
                                    hough_min_line_len=5, hough_max_line_gap=5)
    # print(connected_lines)
    try:
        result1 = draw_connected_lane_lines(np.zeros_like(image), connected_lines)
        result2 = crop_output(result1)
        result3 = Merge_Image(result1, image)
    except Exception as e:
        return np.zeros_like(image)
            
    return result2, result3

def crop_output(image):
    image = image[240:550, 450:1250]
    image = cv2.resize(image, (160, 60))
    return image

<<<<<<< HEAD
lane_model = keras.models.load_model(r'C:\Users\nguye\OneDrive\Desktop\U\kì 6\PBL5\LaneSenseGuard\model_trained\model_trained_v4.h5')
=======

# lane_model = keras.models.load_model(r'C:\Users\nguye\OneDrive\Desktop\U\kì 6\PBL5\LaneSenseGuard\model_trained\model_trained_v4.h5')
lane_model = keras.models.load_model('././model_trained/model_trained_v4.h5')

>>>>>>> f36dc270fe70acc0f887d52985abe8c43cb50b06
dict = {'true': [1, 0], 'false': [0, 1]}
name_result = ['right', 'wrong']
frame_counter = 0
frame_per_predict = 24
<<<<<<< HEAD
cam2 = "http://192.168.145.37:8080/?action=stream"
# cam2 = "http://192.168.137.9:8080/?action=stream"
=======

cam2 = "http://192.168.137.9:8080/?action=stream"
# cam2 = "http://192.168.137.9:8080/?action=stream"


>>>>>>> f36dc270fe70acc0f887d52985abe8c43cb50b06
stream = urllib.request.urlopen(cam2)
bytes = bytes()

import websockets
import asyncio
async def send_and_receive():
    global bytes, frame_counter
<<<<<<< HEAD
    uri = "ws://192.168.145.37:12345"
    async with websockets.connect(uri) as websocket:
=======

    uri = "ws://192.168.137.9:12345"
    uri2 = "ws://103.77.246.238:5001"
    async with websockets.connect(uri) as websocket_1, websockets.connect(uri2) as websocket_2:

>>>>>>> f36dc270fe70acc0f887d52985abe8c43cb50b06
        while True:
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                frame_counter += 1
                if frame_counter == frame_per_predict:
                    frame_counter = 0
                try:
                    image, lane_ft = process_image(i)
                    if frame_counter == 0:   
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        result = name_result[np.argmax(lane_model.predict(image.reshape(-1, 60, 160, 1)))]
                        print(result)
<<<<<<< HEAD
                        await websocket.send("lane:" + result)
                        response = await websocket.recv()
                    vertices = np.array([[(130, 390),(280, 305), (350, 305), (515,390)]], dtype=np.int32) # (480, 640, 3)
                    i = cv2.polylines(i, vertices, isClosed=True, color=(0, 255, 0), thickness=2)
                    # print(i.shape)
=======

                        await websocket_1.send("lane:" + result)
                        response = await websocket_1.recv()
                    vertices = np.array([[(130, 390),(280, 305), (350, 305), (515,390)]], dtype=np.int32) # (480, 640, 3)
                    i = cv2.polylines(i, vertices, isClosed=True, color=(0, 255, 0), thickness=2)
                    # print(i.shape)
                    # Nén ảnh và gửi tới server WebSocket thứ hai
                    i = cv2.resize(i, (320, 240))
                    _, buffer = cv2.imencode('.jpg', i, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                    await websocket_2.send(buffer.tobytes())

>>>>>>> f36dc270fe70acc0f887d52985abe8c43cb50b06
                    cv2.imshow('i', i)
                    
                    if cv2.waitKey(1) == 27:
                        exit(0)
                except Exception as ex:
<<<<<<< HEAD
                    # print(ex)
=======

                    print(ex)

>>>>>>> f36dc270fe70acc0f887d52985abe8c43cb50b06
                    pass
                

asyncio.get_event_loop().run_until_complete(send_and_receive())

# py data-transfer/readStream.py