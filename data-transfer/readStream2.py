
import cv2
import urllib.request
import numpy as np
import dlib
from imutils import face_utils
import os
import uuid
# from dotenv import load_dotenv
from pathlib import Path

# dotenv_path = Path('../../.env')
# load_dotenv(dotenv_path=dotenv_path)

# DATASET_IMAGE_PATH = os.getenv('DATASET_IMAGE_PATH')

# print(DATASET_IMAGE_PATH)

# dirname = os.path.dirname(__file__)

predictor_path = "././drowsiness-detector/models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

cascade_path = "././drowsiness-detector/models/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# detect the face rectangle 
def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    
    # if it doesn't return rectangle return array
    # with zero length
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects

class RightEye:
    def __init__(self, eye):
        self.x_min = eye[:, 0].min() - 4
        self.x_max = eye[:, 0].max() + 20
        self.y_min = eye[:, 1].min() - 22
        self.y_max = eye[:, 1].max()
        self.width = self.x_max - self.x_min
        self.height = int(self.width * (26/34))
        
class LeftEye:
    def __init__(self, eye):
        self.x_min = eye[:, 0].min() - 20
        self.x_max = eye[:, 0].max() + 4
        self.y_min = eye[:, 1].min() - 22
        self.y_max = eye[:, 1].max()
        self.width = self.x_max - self.x_min
        self.height = int(self.width * (26/34))

def getFace(frame, gray):
    def getFirstFace(te):
        if len(te) > 1:
            face = te[0]
            return face
        elif len(te) == 1:
            [face] = te
            return face
	
    te = detect(gray, minimumFeatureSize=(80, 80))
 
    if len(te) == 0:
        return None
    
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # for i, feature in enumerate(te):
    #         x, y, w, h = feature
    #         color = colors[i % len(colors)]  # Pick a color from the list, looping if necessary
    #         cv2.rectangle(frame, (x, y), (w, h), color, 2)


    face = getFirstFace(te)
    
    faceRectangle = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))
    
    # cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (128, 0, 128), 2)
    return faceRectangle

def getEyes(face, gray):
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)
    
    #  grab the indexes of the facial landmarks for the left and
	#  right eye, respectively
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    
    return LeftEye(leftEye), RightEye(rightEye)

def saveImage(frame, eye):
    subfolder_name = "gen_close"
    subfolder_path = os.path.join(DATASET_IMAGE_PATH, subfolder_name)

    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    roi = frame[eye.y_min : eye. y_min + eye.height, eye.x_min : eye.x_min + eye.width]

    roi = cv2.resize(roi, (34, 26))
    
    random_filename = str(uuid.uuid4()) + ".jpg"

    cv2.imwrite(os.path.join(subfolder_path, random_filename), roi)

def main():
    camera = cv2.VideoCapture(0)
	
    while True:
        ret, frame = camera.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face = getFace(frame, gray)
        
        if face is None:
            continue

        leftEye, rightEye = getEyes(face, gray)
        
        cv2.rectangle(frame, (leftEye.x_min, leftEye.y_min ), (leftEye.x_min + leftEye.width, leftEye.y_min + leftEye.height), (0, 255, 0), 2)

        cv2.rectangle(frame, (rightEye.x_min, rightEye.y_min ), (rightEye.x_min + rightEye.width, rightEye.y_min + rightEye.height), (0, 255, 0), 2)
        
        # saveImage(frame, leftEye)
        # saveImage(frame, rightEye)
        
        cv2.imshow('dataset generator', frame)
  
        key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
        if key == ord('q'):
            break
	# do a little clean up
    cv2.destroyAllWindows()
    del(camera)
    
if __name__ == '__main__':
	main()

# cam2 = "http://169.254.142.134:8000/stream.mjpg"
# # cam2 = "http://169.254.142.134:8080/?action=stream"
# stream = urllib.request.urlopen(cam2)
# bytes = bytes()
# while True:
#     bytes += stream.read(1024)
#     a = bytes.find(b'\xff\xd8')
#     b = bytes.find(b'\xff\xd9')
#     if a != -1 and b != -1:
#         jpg = bytes[a:b+2]
#         bytes = bytes[b+2:]
#         i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#         cv2.imshow('i', i)
#         if cv2.waitKey(1) == 27:
#             exit(0)

# py data-transfer/readStream2.py