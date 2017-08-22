import cv2
import sys
import pandas as pd
import numpy as np
import time
#cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier("./OpenCV_FaceCascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)
time.sleep(2)
#classifier
fishface = cv2.face.createFisherFaceRecognizer()

emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise", "neutral"]

fishface.load('./fer2013/fishface_no_fear_disgust.xml')
counter = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    if ret:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]
        #cv2.imshow('test', roi)
            resized_image = cv2.resize(roi, (48, 48)) 
            #histogram normalization
            equ = cv2.equalizeHist(resized_image)
            
            cv2.imwrite("./test_images_saved/frame%d.jpg" % counter, equ)
            
            pred, conf = fishface.predict(equ)
            print counter, pred ,emotions[pred],conf    
    # Display the resulting frame

    cv2.imshow('Video', frame)
    counter+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
