import numpy as np
import cv2
import dlib

Cap = cv2.VideoCapture(0)

feature_params = dict(maxCorners=100,qualityLevel=0.01,minDistance=10,blockSize=3)

lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 68, 0.01))

color = np.random.randint(0,255,(100,3))
p0 = []
Flag = True
old_gray = None
ret = None
old_frame = None
while Flag :
    ret,old_frame = Cap.read()
    old_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    detector_new = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./68Marks.dat")
    dets = detector_new(old_gray,0)
    if len(dets) != 0 :
        Flag = False
    else :
        Flag = True
    if Flag == False :
        print("GOGOGO!")
    for face in dets:
        prediction = predictor(old_gray,face)
        for each in prediction.parts() :
            p0.append([each.x,each.y])
        p0 = np.array(p0)
        # print(list(prediction.parts()))
        # p0 = np.array(prediction.parts())
# p0 = cv2.goodFeaturesToTrack(old_gray,mask=None,**feature_params)
#
# print(type(p0))

while True :
    ret,frame = Cap.read()
    if ret is False :
        break
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    p1,st,err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,p0,None,**lk_params)
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        # frame = cv2.line(frame,(a,b),(c,d),color[i].tolist(),2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)

    old_gray = frame_gray.copy()

    p0 = good_new.reshape(-1,1,2)