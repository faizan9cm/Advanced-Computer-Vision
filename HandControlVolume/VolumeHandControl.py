import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import time
import HandTrackModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

###############################
wCam, hCam = 640, 480
###############################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(detectCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute())
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

pTime = 0
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()

    # Flip the image horizontally
    # img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 

        cv2.circle(img, (x1, y1), 7, (255, 0, 200), cv2.FILLED)
        cv2.circle(img, (x2, y2), 7, (255, 0, 200), cv2.FILLED)
        # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 200), 2)
        cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(length)

        # Hand range 50 - 300
        # Volume range -65 - 0
        vol = np.interp(length, [50, 250], [minVol, maxVol])
        volBar = np.interp(length, [50, 250], [400, 100])
        volPer = np.interp(length, [50, 250], [0, 100])      
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        
        if length < 81:
            cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
        elif length > 221:
            cv2.circle(img, (cx, cy), 7, (0, 255, 255), cv2.FILLED)
    
    # cv2.rectangle(img, (25, 100), (70, 400), (0, 255, 0), 3)
    if volBar <= 150:
        cv2.rectangle(img, (25, int(volBar)), (70, 400), (0, 255, 255), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (24, 440),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
    elif volBar > 350:
        cv2.rectangle(img, (25, int(volBar)), (70, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (24, 440),cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
    else:
        cv2.rectangle(img, (25, int(volBar)), (70, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)}%', (24, 440),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    # cv2.putText(img, f'{int(volPer)}%', (24, 440),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 50), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 25),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 50), 3)

    cv2.imshow("Volumne Control", img)
    cv2.waitKey(1)