import numpy as np
import cv2

a = np.arange(0,6).reshape(2,3)
img = cv2.imread('test_pic.jpeg')
print(img.shape)
cv2.imshow('img_real',img)
a = img[90:180,200:260]
img[30:120,60:120] = a
b,g,r = cv2.split(img)
cv2.imshow('blue',b)
cv2.imshow('green',g)
cv2.imshow('red',r)
cv2.addWeighted()
img_new = np.dstack([g,g,g])
cv2.imshow('merge',img_new)
cv2.waitKey(0)