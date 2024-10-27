import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import time
import HandTrackModule as htm
import numpy as np
import tkinter as tk
import pyautogui
pyautogui.FAILSAFE = False

wCam, hCam = 640, 480
frameR = 150 # Frame reduction
smoothening = 1.5
scaling_factor = 1.5  # Adjust this value to control sensitivity

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

root = tk.Tk()
wScr = root.winfo_screenwidth()
hScr = root.winfo_screenheight()
# print(wScr, hScr)

detector = htm.handDetector(maxHands=1)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

frame_skip = 1 # Process every 2nd frame
frame_count = 0

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Only process every 2nd frame for performance
    if frame_count % frame_skip == 0:
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        # print(lmList)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # print(x1, y1, x2, y2)

        fingers = detector.fingersUp()
        # print(fingers)

        # Adjust the rectangle
        moveXL, moveYT, moveXR, moveYB = 40, 80, 40, 100
        
        cv2.rectangle(img, (frameR+moveXL, frameR-moveYT), (wCam-frameR-moveXR, hCam-frameR-moveYB), (255, 0, 255), 2)

        # Move mode
        if len(fingers) != 0 and fingers[1] == 1 and fingers[2] == 0:
            # Check if the finger is inside the rectangle
            if (frameR + moveXL - 20 < x1 < wCam - frameR - moveXR + 20) and (frameR - moveYT - 20 < y1 < hCam - frameR - moveYB + 20):
                # Interpolating screen coordinates 
                # x3 = np.interp(x1, (0, wCam), (0, wScr))
                # y3 = np.interp(y1, (0, hCam), (0, hScr))
                x3 = np.interp(x1, (frameR + moveXL, wCam - frameR - moveXR), (0, wScr))
                y3 = np.interp(y1, (frameR - moveYT, hCam - frameR - moveYB), (0, hScr))

                # Smoothen values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                # pyautogui.moveTo(x3, y3)
                pyautogui.moveTo(clocX, clocY)
                cv2.circle(img, (x1, y1), 3, (0, 255, 0), cv2.FILLED)
                plocX, plocY = clocX, clocY

        # Click mode
        if len(fingers) != 0 and fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (25, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0 ,255), 3)

    cv2.imshow("AI Virtual Mouse", img)
    cv2.waitKey(1)

root.destroy()
cap.release()
cv2.destroyAllWindows()