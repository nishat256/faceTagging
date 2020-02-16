import numpy as np 
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
	"""
		This function is used to detect faces in image provided as input and return list of detected faces.
	"""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.5, 5)
	faces_detected = []
	for (x,y,w,h) in faces:
		# To draw a rectangle in a face  
		#cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
		roi_gray = gray[y:y+h, x:x+w] 
		roi_color = img[y:y+h, x:x+w]
		roi_color = cv2.resize(roi_color,(224,224))
		faces_detected.append(roi_color)
	return faces_detected
if __name__ == "__main__":
	resp = detect_face(cv2.imread('input/image2.jpg'))
	cv2.imwrite('faces/salman2.jpg',resp[1])
