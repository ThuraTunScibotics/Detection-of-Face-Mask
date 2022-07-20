from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import numpy as np
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab dimension of frame
    (h, w) = frame.shape[:2]
    
    # construct a blob from the input frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 117.0, 123.0))
    
    # forward pass inputBlob to FaceNet
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # initialize the predictions requirements
    faces = []
    locs = []
    preds = []
    
    # loop over the detection
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # if confidence is greater than probability 0.5, get bbox locations
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    
    # if number of person is more than at least 1
    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        preds = maskNet.predict(faces)
        
    return (locs, preds)



# load face & mask detector model from disk
print("[INFO]: Loading face detector model from disk...")
protoPath = '/home/thura/Desktop/TSF-internship/detection-of-face-mask/face-detect-model/deploy.prototxt'
weightPath = '/home/thura/Desktop/TSF-internship/detection-of-face-mask/face-detect-model/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(protoPath, weightPath)

print("[INFO]: Loading mask detector model from disk...")
maskModel = '/home/thura/Desktop/TSF-internship/detection-of-face-mask/mask-detector-model.model'
maskNet = load_model(maskModel)


# Start Video Streaming from WecCam
print("[INFO]: Start Video streaming....")
vs = VideoStream(src=2).start()
# vs = cv2.VideoCapture(2)
time.sleep(2.0)


# Loop over the frame from video stream...
while True:
    frame = vs.read()
    
    frame = imutils.resize(frame, width=400)
        
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        
    # loop over the detected face locations and their corresponding predictions
    for (bbox, preds) in zip(locs, preds):
        # unpack bounding-box and predictions
        (startX, startY, endX, endY) = bbox
        (mask, withoutMask) = preds
            
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label=="Mask" else (0, 0, 255)
            
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            
        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(10) & 0xFF
    
    if key == ord('q'):
        break
        
cv2.destroyAllWindows()
vs.stop()
# vs.release()