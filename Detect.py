# from picamera.array import PiRGBArray
# from picamera import PiCamera
import time
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# from google.colab import files
import numpy as np
import tarfile
import shutil
import json
import math
import cv2
import os
import numpy
from PIL import Image

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
        
        y1 = imshape[0] # bottom of the image
        y2 = y1*0.6         # slightly lower than the middle

        left_line  = make_line_points(y1, y2, left_lane)
        right_line = make_line_points(y1, y2, right_lane)
        
        connected_lines = np.int32([[left_line, right_line]])
        
        return connected_lines # left_line, right_line
    except Exception as e:
        print(e)

def Detect_Lane(image, plot_image=False,
                kernel_size=5, canny_low_threshold=50, canny_high_threshold=150,
                hough_rho=1, hough_theta=np.pi/180, hough_threshold=20,
                hough_min_line_len=20, hough_max_line_gap=300):
    
    # filter to select those white and yellow lines
    # white_yellow_image = select_white_yellow(image)
    
    # Convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray_blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Canny Edge Detection
    edges = cv2.Canny(gray_blur, canny_low_threshold, canny_high_threshold)

    # Concentrate the location of edge detection
    image_shape = image.shape
    # vertices = vertices = np.array([[(100,image_shape[0]),
    #                       (image_shape[1]/2-30, image_shape[0]/2+50),
    #                       (image_shape[1]/2+30, image_shape[0]/2+50),
    #                       (image_shape[1]-20, image_shape[0])]], dtype=np.int32)
    # vertices = np.array([[(535, 415), (1185, 415), (822, 280), (788, 280)]], dtype=np.int32)
    
    vertices = np.array([[(450,image_shape[0]-50),(700, 400), (800, 400), (1150,image_shape[0]-50)]], dtype=np.int32)
    
    masked_edges, mask = Get_Lane_Area(edges, vertices=vertices)

    # Detect lines using Hough transform on an edge detected image
    lines_image, lines = Hough_Transformation(masked_edges,
                                  rho=hough_rho, theta=hough_theta, threshold=hough_threshold,
                                  min_line_len=hough_min_line_len, max_line_gap=hough_max_line_gap)

    # Merge 'original' image with 'lines' image
    result = Merge_Image(lines_image, image)
    connected_lines = connect_lane_lines(lines, image_shape)
    
    # Plot the images
    if plot_image:
        plt.figure(figsize=[16, 9])
        for i, img in enumerate(['gray', 'gray_blur', 'edges', 'mask', 'masked_edges', 'lines_image', 'result']):
            Plot_Image(eval(img), img, (4,2, i+1))
            plt.axis('off')
            plt.show()
    return result, connected_lines, gray, gray_blur, edges, masked_edges, lines_image

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
    
    # vertices = []
    # l_line, r_line = connected_lines[0][0], connected_lines[0][1]

    # vertices.append((l_line[0], l_line[1]))
    # vertices.append((l_line[2], l_line[3]))
    # vertices.append((r_line[2], l_line[3]))
    # vertices.append((r_line[0], r_line[1]))
    
    # cv2.fillPoly(lines_img, np.int32([vertices]), [0,255,0])
    
    # Perform image blending
    return Merge_Image(lines_img, img)

def process_image(image):
    
    # Find the lanes
    result, connected_lines, gray, gray_blur, edges, mask, lines_image = Detect_Lane(image, plot_image=False,
                                    kernel_size=5, canny_low_threshold=50, canny_high_threshold=150,
                                    hough_rho=1, hough_theta=np.pi/180, hough_threshold=20,
                                    hough_min_line_len=5, hough_max_line_gap=5)
    # print(connected_lines)
    try:
        result1 = draw_connected_lane_lines(np.zeros_like(image), connected_lines)
        result2 = crop_output(result1)
        # for i, img in enumerate(['image', 'gray', 'gray_blur', 'edges', 'mask', 'lines_image']):
        #     Plot_Image(eval(img), img, (4,2, i+1))
        #     plt.axis('off')
        #     plt.show()
    except Exception as e:
        # for i, img in enumerate(['image', 'gray', 'gray_blur', 'edges', 'mask', 'lines_image']):
        #     Plot_Image(eval(img), img, (4,2, i+1))
        #     plt.axis('off')
        #     plt.show()
        return None, result, lines_image
            
    return connected_lines, result2, lines_image
def crop_output(image):
    image = image[240:550, 450:1250]
    image = cv2.resize(image, (160, 60))
    return image

modelTrained = keras.models.load_model('./model_trained.h5')
# initialize the camera and grab a reference to the raw camera capture
# camera = PiCamera()
# camera.resolution = (640, 480)
# camera.framerate = 24
# rawCapture = PiRGBArray(camera, size=(640, 480))
# #Load a cascade file for detecting faces
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# print ("\n [INFO] Initializing face capture. Look the camera and wait ...")
# # capture frames from the camera
# for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
#     # convert frame to array
#     image = frame.array
#     #Convert to grayscale
#     gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     #Look for faces in the image using the loaded cascade file
#     faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5, minSize = (100, 100), flags = cv2.CASCADE_SCALE_IMAGE)
 
#     print ("Found "+str(len(faces))+" face(s)")
#     #Draw a rectangle around every found face
#     for (x,y,w,h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
#         print(x,y,w,h)
#     # display a frame    
#     cv2.imshow("Frame", image)
#     #wait for 'q' key was pressed and break from the loop
#     if cv2.waitKey(1) & 0xff == ord("q"):
#         exit()
#     # clear the stream in preparation for the next frame
#     rawCapture.truncate(0)

# modelTrained.summary()

# dự đoán
# predict = modelTrained.predict(numpy.array(Xval))
width = 160
height = 60
# img = cv2.imread('./lane/04950.jpg')
# print(img.shape)

# connected_lines, result, lines_image = process_image(img)
# result = cv2.resize(result, (width, height))
name_result = ['right', 'wrong']
# def pr(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     result = name_result[numpy.argmax(modelTrained.predict(image.reshape(-1, height, width, 1)))]
#     return result

# print(pr(result))


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 24
rawCapture = PiRGBArray(camera, size=(640, 480))
#Load a cascade file for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print ("\n [INFO] Initializing face capture. Look the camera and wait ...")
# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # convert frame to array
    image = frame.array
    #Convert to grayscale

    connected_lines, result, lines_image = process_image(image)
    result = cv2.resize(result, (width, height))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = name_result[numpy.argmax(modelTrained.predict(image.reshape(-1, height, width, 1)))]
    print(result)

    #Draw a rectangle around every found face
    # display a frame    
    cv2.imshow("Frame", image)
    #wait for 'q' key was pressed and break from the loop
    if cv2.waitKey(1) & 0xff == ord("q"):
        exit()
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)