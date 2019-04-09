import cv2
import numpy as np

image = cv2.imread('./test.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(
   gray,
   scaleFactor = 1.15,
   minNeighbors = 5,
   minSize = (5,5)
)
print ("发现{0}个人脸!".format(len(faces)))

for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)

cv2.imshow('test.jpg',image)
cv2.waitKey(0)