import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import time
import HandTrackModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

detector = htm.handDetector(detectCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw = False)
    print(lmlist)

    # Fliz image horizontally
    img = cv2.flip(img, 1)

    if len(lmlist) != 0:
        fingers = []

        # Thumb
        if lmlist[tipIds[0]][1] < lmlist[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)

        totalFingers = fingers.count(1)
        print(totalFingers)

        cv2.rectangle(img, (20, 50), (150, 250), (0, 255, 150), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (35, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 7)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 25), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv2.imshow("Finger Counter", img)
    cv2.waitKey(1)