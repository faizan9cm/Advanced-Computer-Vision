import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import time
import numpy as np
import PoseEstimateModule as pem

cap = cv2.VideoCapture(0)
detector = pem.poseDetector()

pTime = 0
count = 0
dir = 0
per = 0
bar = 430
color = (255, 0, 255)

while True:
    success, img = cap.read()
    
    # Flip the image horizontally
    img = cv2.flip(img, 1)
    # img = cv2.resize(img, (1280, 720))

    img = detector.findPose(img, draw=False)
    lmlist = detector.findPosition(img, draw=False)

    if len(lmlist) != 0:
        # detector.findAngle(img, 12, 14, 16)
        angle = detector.findAngle(img, 11, 13, 15)

        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (210, 310), (430, 90))
        # print(angle, per)

        # Check for the dumbell curls
        if per == 100:
            color = (0, 255, 0)
            if dir == 0:
                count += 0.5
                dir = 1
        
        if per == 0:
            color = (255, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0

        # print(count)

    cv2.rectangle(img, (0, 320), (160, 480), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (30, 455), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 7)

    cv2.rectangle(img, (575, 90), (610, 430), color, 2)
    cv2.rectangle(img, (575, int(bar)), (610, 430), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)}%', (570, 80), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)

    cTime = time.time()
    
    fps = 1 / (cTime - pTime) 
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv2.imshow("AI Trainer", img)
    cv2.waitKey(1)