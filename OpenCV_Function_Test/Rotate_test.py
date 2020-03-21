import cv2
import numpy as np
import dlib
import math


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
Left_Arrows = Points[17:22]
Right_Arrows = Points[22:27]
Left_Arrow_Center = np.mean(Left_Arrows,axis=0).astype("int")
Right_Arrow_Center = np.mean(Right_Arrows,axis=0).astype("int")


Left_Arrow_Center = tuple(Left_Arrow_Center)
Right_Arrow_Center = tuple(Right_Arrow_Center)

cv2.circle(img,Left_Arrow_Center,2,(0,255,0),2)
cv2.circle(img,Right_Arrow_Center,2,(0,255,0),2)

cv2.imshow("img",img)


# 图像整体旋转 :
# 计算旋转参数
Center_X = (Left_Arrow_Center[0]+Right_Arrow_Center[0])/2
Center_Y = (Left_Arrow_Center[1]+Right_Arrow_Center[1])/2
center = (int(Center_X),int(Center_Y))
# 旋转中心
dy = Right_Arrow_Center[1] - Left_Arrow_Center[1]
dx = Right_Arrow_Center[0] - Left_Arrow_Center[0]
angle = math.atan2(dy,dx) * 180. / math.pi
# 旋转角度 单位弧度
rotate_matrix = cv2.getRotationMatrix2D(center,angle,scale=1)
# 获得旋转矩阵 输入参数 : 中心点 center 旋转角度 angle 图像大小 scale = (Height,Width)
rotated_img = cv2.warpAffine(img,rotate_matrix,(img.shape[1],img.shape[0]))


# 关键点坐标变换 :
def rotate_single(origin,point,angle,row):
    x1,y1 = point
    x2,y2 = origin
    y1 = row-y1
    y2 = row-y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle)*(x1-x2)-math.sin(angle)*(y1-y2)
    y = y2 + math.sin(angle)*(x1-x2)+math.cos(angle)*(y1-y2)
    y = row-y
    return int(x),int(y)

Rotated_Points = []
for each in Points:
    X,Y = rotate_single((Center_X,Center_Y),tuple(each),angle,img.shape[0])
    Rotated_Points.append((X,Y))

for each in Rotated_Points :
    cv2.circle(rotated_img,each,2,[255,0,0],1)

count = 0
for each in Rotated_Points :
    cv2.putText(rotated_img,"{}".format(count),(each[0],each[1]),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,255),1)
    count = count + 1

Rotated_Points_List = Rotated_Points
Rotated_Points = np.array(Rotated_Points)
#获得变换后左右眉毛均值中心点的坐标:
Rotated_Arrow_Left_Center = np.mean(Rotated_Points[17:22],axis=0).astype("int")
Rotated_Arrow_Right_Center = np.mean(Rotated_Points[22:27],axis=0).astype("int")
Rotated_Nose_Bottom_Center = np.mean(Rotated_Points[31:36],axis=0).astype("int")

Face_Shaped = rotated_img[Rotated_Points[19][1]:Rotated_Points[33][1],Rotated_Points[17][0]:Rotated_Points[26][0]]
mabbb = np.array([[200,300],[300,400]]).astype("int")
Face_Resize = cv2.resize(Face_Shaped,(200,200),interpolation=cv2.INTER_CUBIC)
cv2.imshow("File",rotated_img)
cv2.imshow("face",Face_Shaped)
cv2.imshow("Face_Reshape",Face_Resize)


cv2.waitKey(0)