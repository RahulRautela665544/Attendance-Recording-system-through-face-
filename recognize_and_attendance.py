import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import os
import time

# Load embeddings
data = np.load("embeddings/faces_embeddings.npz")
X = data["X"]
y_labels = data["y"]

detector = MTCNN()
embedder = FaceNet()

os.makedirs("attendance", exist_ok=True)

cam = cv2.VideoCapture(0)

print("Secure Face Attendance Running... Press ESC to exit")

marked_names = set()

# Liveness state
liveness_stage = 0
left_verified = False
right_verified = False
last_verified_time = 0


def reset_liveness():
    global liveness_stage, left_verified, right_verified
    liveness_stage = 0
    left_verified = False
    right_verified = False


while True:

    ret, frame = cam.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    # -------------------------
    # NO FACE
    # -------------------------
    if len(faces) == 0:

        reset_liveness()

        cv2.putText(frame,
                    "No Face Detected",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        cv2.imshow("Secure Face Attendance", frame)

        if cv2.waitKey(1) == 27:
            break

        continue

    # -------------------------
    # MULTIPLE FACES
    # -------------------------
    if len(faces) > 1:

        reset_liveness()

        cv2.putText(frame,
                    "ONLY ONE PERSON ALLOWED",
                    (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        cv2.putText(frame,
                    "Others move out of frame",
                    (30,90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2)

        for f in faces:
            x, y, w, h = f['box']
            x, y = abs(x), abs(y)

            cv2.rectangle(frame,
                          (x,y),
                          (x+w,y+h),
                          (0,0,255),
                          3)

        cv2.imshow("Secure Face Attendance", frame)

        if cv2.waitKey(1) == 27:
            break

        continue

    # -------------------------
    # EXACTLY ONE FACE
    # -------------------------
    face = faces[0]

    x, y, w, h = face['box']
    x, y = abs(x), abs(y)

    face_crop = frame[y:y+h, x:x+w]

    if face_crop.size == 0:
        continue

    face_center = x + w // 2
    frame_center = frame.shape[1] // 2
    offset = face_center - frame_center

    # -------------------------
    # Stage 0 → Turn Left
    # -------------------------
    if liveness_stage == 0:

        cv2.putText(frame,
                    "Turn Head LEFT",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        if offset < -60:
            left_verified = True
            liveness_stage = 1

    # -------------------------
    # Stage 1 → Turn Right
    # -------------------------
    elif liveness_stage == 1:

        cv2.putText(frame,
                    "Turn Head RIGHT",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        if offset > 60:
            right_verified = True
            liveness_stage = 2
            last_verified_time = time.time()

    # -------------------------
    # Stage 2 → Liveness Verified
    # -------------------------
    elif liveness_stage == 2 and left_verified and right_verified:

        cv2.putText(frame,
                    "Liveness Verified",
                    (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    3)

        if time.time() - last_verified_time > 5:
            reset_liveness()
            continue

        face_img = cv2.resize(rgb[y:y+h, x:x+w], (160,160))

        emb = embedder.embeddings([face_img])[0]

        scores = cosine_similarity([emb], X)[0]

        idx = np.argmax(scores)

        if scores[idx] > 0.7:

            name = y_labels[idx]

            if name not in marked_names:

                time_now = datetime.now().strftime("%H:%M:%S")

                df = pd.DataFrame([[name,time_now]],
                                  columns=["Name","Time"])

                df.to_csv("attendance/attendance.csv",
                          mode='a',
                          header=not os.path.exists("attendance/attendance.csv"),
                          index=False)

                marked_names.add(name)

            cv2.rectangle(frame,
                          (x,y),
                          (x+w,y+h),
                          (0,255,0),
                          2)

            cv2.putText(frame,
                        name,
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

        else:

            cv2.rectangle(frame,
                          (x,y),
                          (x+w,y+h),
                          (255,0,0),
                          2)

            cv2.putText(frame,
                        "Unknown",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        2)

    cv2.imshow("Secure Face Attendance", frame)

    if cv2.waitKey(1) == 27:
        break


cam.release()
cv2.destroyAllWindows()