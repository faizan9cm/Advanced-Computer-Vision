import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectCon = minDetectCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.landmark_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 124))
        self.connection_spec = self.mpDraw.DrawingSpec(thickness=1, color=(0, 255, 124))
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectCon,
            min_tracking_confidence=self.minTrackCon
            )

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLMS, self.mpFaceMesh.FACEMESH_CONTOURS, self.landmark_spec, self.connection_spec)
                face = []
                for id, lm in enumerate(faceLMS.landmark):
                    h, w, c = img.shape
                    x, y, = int(lm.x*w), int(lm.y*h)
                    # print(id, x, y)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

                    face.append([x, y])
                faces.append(face)
        return img, faces   


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)

        # if len(faces) != 0:
            # print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 25), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cv2.imshow("Face Mesh", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()