import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet

detector = MTCNN()
embedder = FaceNet()

X = []
y = []

for person in os.listdir("dataset"):
    person_path = os.path.join("dataset", person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        if faces:
            x, y1, w, h = faces[0]['box']
            face = rgb[y1:y1+h, x:x+w]
            face = cv2.resize(face, (160, 160))

            embedding = embedder.embeddings([face])[0]
            X.append(embedding)
            y.append(person)

np.savez("embeddings/faces_embeddings.npz", X=X, y=y)
print("Face embeddings generated and saved.")
