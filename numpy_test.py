import numpy as np
import cv2

img = cv2.imread('test_pic.jpeg')
ball=img[280:340,330:390]
img[273:333,100:160]=ball

cv2.imshow('img',img)
cv2.waitKey(0)



# Width_Pixel = 1920
# Height_Pixel = 1080
# img = np.zeros((Width_Pixel,Height_Pixel,3),np.uint8)
# #创建一个空图像
#
# img[200,100] = [255,255,255]
# #改变某个像素点的颜色
#
# print(img.shape) ## ( 1920 , 1080 , 3 )
#  ##输出一个元组,表明其图像通道数
#
# print(img.size) ##  1920 x 1080  = 2073600
#  ##输出图像的像素数目
#
# print(img.dtype)