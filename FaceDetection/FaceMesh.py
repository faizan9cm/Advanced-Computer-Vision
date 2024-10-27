import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpDraw = mp.solutions.drawing_utils
landmark_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(26, 255, 0))
connection_spec = mpDraw.DrawingSpec(thickness=1, color=(50, 50, 255))
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
# faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2, refine_landmarks=True)

pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLMS in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLMS, mpFaceMesh.FACEMESH_TESSELATION, landmark_spec, connection_spec)

            for id, lm in enumerate(faceLMS.landmark):
                # print(lm)
                h, w, c = img.shape
                x, y = int(lm.x*w), int(lm.y*h)
                print(id, x, y)


            # left_iris = [faceLMS.landmark[i] for i in range(468, 473)]
            # right_iris = [faceLMS.landmark[i] for i in range(473, 478)]

            # Convert landmarks to pixel coordinates
            # h, w, _ = img.shape
            # left_center = [int(sum([p.x for p in left_iris]) / len(left_iris) * w),
            #                int(sum([p.y for p in left_iris]) / len(left_iris) * h)]
            # right_center = [int(sum([p.x for p in right_iris]) / len(right_iris) * w),
            #                 int(sum([p.y for p in right_iris]) / len(right_iris) * h)]

            # Draw the pupils (center of the irises)
            # cv2.circle(img, tuple(left_center), 3, (0, 255, 0), -1)  # Left pupil
            # cv2.circle(img, tuple(right_center), 3, (0, 255, 0), -1)  # Right pupil
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FSP: {int(fps)}', (20, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 50), 2)

    cv2.imshow("Face Mesh", img)
    cv2.waitKey(1)


           