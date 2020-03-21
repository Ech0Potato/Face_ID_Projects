import numpy as npy
import cv2
import base64

img = npy.zeros((512,512,3),npy.uint8)
cv2.line(img,(2,2),(400,400),(255,0,200),3,lineType = cv2.LINE_AA)
cv2.circle(img,(100,100),50,(255,0,30),thickness = 2 , lineType= cv2.LINE_AA)
cv2.rectangle(img,(20,20),(400,400),thickness = 5 , color = (255,0,0),lineType= cv2.LINE_AA)
cv2.ellipse(img,(240,240),(80,30),117,0,355,color = (255,3,2),thickness = 2, lineType= cv2.LINE_AA)
pts = npy.array([[300,200],[200,150],[450,220],[210,310]],npy.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,0),thickness= 2)
f = open('img.png','rb')
ls_f = base64.b64encode(f.read())
f.close()
print(ls_f)
f = open('./base64','rb')
f.write(ls_f)
f.close()
cv2.imshow('img',img)
cv2.imwrite('img.png',img)

cv2.waitKey(0)