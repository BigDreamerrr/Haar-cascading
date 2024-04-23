import cv2
import os

img_path = r"D:\Computer vision\Images\train_faces\non-face\archive (3)"
store_path = r"D:\Computer vision\Images\train_faces\non-face\images\random_subwindow"

for dirpath, dnames, fnames in os.walk(img_path):
    for f in fnames[:135]:
        full_path = os.path.join(dirpath, f)
        img = cv2.imread(full_path)

        cnt = 0
        for i in range(0, img.shape[0] - 70, 50):
            for j in range(0, img.shape[1] - 70, 50):
                cv2.imwrite(f"{store_path}\\{cnt}_{f}", img[i:i+70, j:j+70])
                cnt += 1