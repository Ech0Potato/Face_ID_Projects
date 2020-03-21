import cv2
import numpy as np
import math

def rotate_fit(img,landmarks):
    # 变量重命名,因为懒得改代码


    # 提取左右眉毛中心点 -------------

    # 左眉毛索引起点与终点(图片的左右,不是人的左右眼)
    LEFT_EYEBROW_START = 17
    LEFT_EYEBROW_END = 22

    # 右眉毛索引起点与终点(图片的左右,不是人的左右眼)
    RIGHT_EYEBROW_START = 22
    RIGHT_EYEBROW_END = 27

    # 取出左右眼眉毛点集合
    left_arrow_points = landmarks[LEFT_EYEBROW_START:LEFT_EYEBROW_END]
    right_arrow_points = landmarks[RIGHT_EYEBROW_START:RIGHT_EYEBROW_END]

    # 取集合平均点
    left_arrow_center = np.mean(left_arrow_points,axis=0).astype("int")
    right_arrow_center = np.mean(right_arrow_points,axis=0).astype("int")


    # 旋转参数准备 -------------------

    # 设置旋转中心坐标
    rotate_center = np.mean([left_arrow_center,right_arrow_center],axis=0).astype("int")

    # 计算旋转角度
    delta_x = right_arrow_center[0] - left_arrow_center[0]
    delta_y = right_arrow_center[1] - left_arrow_center[1]
    rotate_angle_rad = math.atan2(delta_y,delta_x)
    rotate_angle_deg = rotate_angle_rad / math.pi * 180.0

    # 确定仿射变换矩阵
    rotate_matrix = cv2.getRotationMatrix2D(tuple(rotate_center),rotate_angle_deg,scale=1)

    # 仿射变换
    rotate_img = cv2.warpAffine(img,rotate_matrix,(img.shape[1],img.shape[0]))

    # 关键点坐标变换函数
    def rotate_single_point(center,point,angle_rad,row_pixels):
        x1, y1 = point[0],point[1]
        x2, y2 = center[0],center[1]
        y1 = row_pixels - y1
        y2 = row_pixels - y2
        angle = angle_rad
        x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
        y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
        y = row_pixels - y
        return int(x),int(y)

    landmarks_rotated = list()
    for each_landmark in landmarks :
        each_landmark_rotated = rotate_single_point(rotate_center,each_landmark,rotate_angle_rad,img.shape[0])
        landmarks_rotated.append(each_landmark_rotated)
    landmarks_rotated = np.array(landmarks_rotated)

    landmarks_nose_bottom_center = np.mean(landmarks_rotated[31:36],axis=0).astype("int")
    landmarks_left_eyebrow_center = np.mean(landmarks_rotated[LEFT_EYEBROW_START:LEFT_EYEBROW_END],axis=0).astype("int")
    landmarks_right_eyebrow_center = np.mean(landmarks_rotated[RIGHT_EYEBROW_START:RIGHT_EYEBROW_END],axis=0).astype("int")

    # face_shaped = rotate_img[landmarks_left_eyebrow_center[1]:landmarks_nose_bottom_center[1],
    #               landmarks_left_eyebrow_center[0]:landmarks_right_eyebrow_center[0]]
    face_shaped = rotate_img[landmarks_rotated[19][1]:landmarks_rotated[33][1],
                             landmarks_rotated[17][0]:landmarks_rotated[26][0]]

    return face_shaped



    

