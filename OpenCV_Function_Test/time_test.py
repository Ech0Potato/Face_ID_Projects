import time
import cv2
import numpy as np
import math

count = 0
cap = cv2.VideoCapture(0)
r = cap.read()
r = cap.read()
time_count = list()

for t in range(50):
    print(t)
    t1 = time.time()
    for i in range(t+1):
        r = cap.read()
    t2 = time.time()
    time_count.append(t2-t1)

for i in range(50):
    time_count[i] = time_count[i] / (i+1)

for i in time_count :
    print(i)
print(np.var(time_count))