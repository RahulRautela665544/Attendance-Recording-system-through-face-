import cv2
import os

name = input("Enter student name: ")
path = os.path.join("dataset", name)
os.makedirs(path, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0

print("Capturing images... Press ESC to stop")

while count < 30:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)
    cv2.imwrite(os.path.join(path, f"{count}.jpg"), frame)
    count += 1

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
print("Images saved in", path)
