import numpy as np
import cv2
import dlib
import time
from imutils import face_utils
from RoFit import rotate_fit


# 角点检测参数
while True:
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=10, blockSize=3)

    # KLT光流参数
    lk_params = dict(winSize=(101, 101), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.001))

    # 随机颜色
    color = np.random.randint(0,255,(100,3))

    cap = cv2.VideoCapture("stable.mp4")
    cap.set(3,1280)
    cap.set(4,720)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./68Marks.dat")
    p0 = []
    Face_Counter = 0
    old_gray = None

    aligner = face_utils.FaceAligner(predictor,desiredFaceWidth=512,desiredFaceHeight=512,desiredLeftEye=(0.30,0.30))
    while True :
        rect,frame = cap.read()
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        old_gray = frame_gray
        print(old_gray.shape)
        dets = detector(frame_gray,0)
        if len(dets) != 0 :
            face = dets[0]
            Face_Counter = Face_Counter + 1
            left = face.left()
            right = face.right()
            bottom = face.bottom()
            top = face.top()
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
            prediction = predictor(frame_gray,face)
            if Face_Counter == 3 :
                for each in prediction.parts():
                    p0.append([each.x, each.y])
                p0 = np.array(p0, dtype=np.float32)
                break
        cv2.imshow("frame",frame)
        cv2.waitKey(1)
    print("Break!")
    # cv2.imshow("OldGray!",old_gray)
    cv2.waitKey(1000)
    while True:
        ret, frame = cap.read()
        frame_backup = frame
        print(ret)
        if ret is False:
            print("bad ret")
            break
        print("Here 3!")
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算光流
        print("Here 4!")
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # 根据状态选择
        print(p1.shape)
        good_new = p1
        good_old = p0
        print("Here 1 !")
        # 绘制跟踪线
        count_pixel = 0
        for i, (new, old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            # frame = cv2.line(frame, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),3,color[i].tolist(),2)
            cv2.putText(frame,"{0}".format(count_pixel),(a,b),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,255),1)
            count_pixel = count_pixel + 1
        rotate_img = rotate_fit(frame_backup,p1.reshape(68,2))
        print("Shape:",rotate_img.shape,"比值",rotate_img.shape[1]/rotate_img.shape[0])
        rotate_img = cv2.resize(rotate_img,(600,400),interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Pick up',rotate_img)
        print("Here 2 !")
        cv2.imshow('Frame',frame)
        cv2.imwrite("Frame_Points.jpg", frame)
        k = cv2.waitKey(33) & 0xff
        if k == 27:
            break

        # 更新
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()