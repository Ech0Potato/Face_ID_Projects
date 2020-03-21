import cv2
import numpy as np
import dlib
import math
from RoFit import rotate_fit

file_name = "Face.jpg"

img = cv2.imread(file_name)

#人脸识别 + 人脸特征点检测
detector = dlib.get_frontal_face_detector()
predictior = dlib.shape_predictor("../68Marks.dat")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dets = detector(img_gray,0)
face = dets[0]
face_points = predictior(img_gray,face)
Points = list()
for each in face_points.parts():
    Points.append([each.x,each.y])
Points = np.array(Points)

img_rotate = rotate_fit(img,Points)

cv2.imshow("Test",img_rotate)
cv2.waitKey(0)