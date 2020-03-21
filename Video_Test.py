import cv2
import dlib
import time
import numpy as np
from imutils import face_utils
import imutils

# # 获取编号0摄像头的捕捉对象
# Cap = cv2.VideoCapture(0)
# # 设置摄像头捕捉参数 3代表每一帧的宽 Width
# Cap.set(3, 1920)
# # 设置摄像头捕捉参数 4代表每一帧的高 Height
# Cap.set(4, 1080)

Cap = cv2.VideoCapture("changrui.mp4")
Cap.set(3,1080)
Cap.set(4,1920)
flag = 1  # 设置标志，表明是否输出视频信息

Write_fourcc = cv2.VideoWriter_fourcc(*"DIVX")
Write_wending = cv2.VideoWriter('Wending_AllPic_yidong.avi',Write_fourcc,30.0,(1920,1080))
Write_Face = cv2.VideoWriter('Wending_SingleFace_yidong.avi',Write_fourcc,30.0,(512,512))

detector_new = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./68Marks.dat")
aligner = face_utils.FaceAligner(predictor,desiredFaceWidth=512,desiredFaceHeight=512,desiredLeftEye=(0.30,0.30))

# 主循环监测视频流状态 如果True，则视频流已经打开
while (Cap.isOpened()):
    ret_bool, frame = Cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    t1 = time.time()
    dets = detector_new(gray, 0)
    print('计算时间:',time.time()-t1)
    # print(type(dets))
    for face in dets:
        confidence = face.confidence
        face = face.rect
        print('置信度:', confidence)
        prediction = predictor(gray,face)
        print(type(prediction))
        face_frame = aligner.align(frame,gray,face)
        cv2.imshow('Face',face_frame)
        Write_Face.write(face_frame)
        print(face_frame.shape)
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        for each in prediction.parts() :
            cv2.circle(frame,(each.x,each.y),3,[0,0,255],2)
    t2 = time.time()
    cv2.imshow('Capture_Test', frame)
    Write_wending.write(frame)
    k = cv2.waitKey(1)
Write.release()