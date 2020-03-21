import numpy as np
import cv2

Cap = cv2.VideoCapture(0)
Cap.set(3,800)
Cap.set(4,600)
while (Cap.isOpened()):
    # filename = "test_pic.jpeg"
    # img = cv2.imread(filename)
    # img2 = img

    flag , img = Cap.read()
    img2 = img

    img_green = np.array(img2[:,:,1])
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray)

    good_features_corners = cv2.goodFeaturesToTrack(img_gray,25,0.01,10)
    good_features_corners = np.int0(good_features_corners)

    for i in good_features_corners:
        x,y = i.flatten()
        cv2.circle(img2,(x,y),3,[0,255,0],-1)

    cv2.imshow('good',img2)
    cv2.imshow('Green',img_green)

    cv2.waitKey(1)


