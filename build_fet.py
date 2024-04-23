import os
os.add_dll_directory(r"D:\Computer vision\Adaboost")

import sys
import numpy as np
from c_adaboost import SimpleAdaBoost, haar
from c_adaboost import LEFT_RIGHT, ABOVE_BELOW, DIAGONAL, LINE
import cv2
from sklearn.preprocessing import scale

standard_size= (131, 170)
img_path = r"D:\Computer vision\Images\train_faces\non-face\images"
fet_path = r"D:\Computer vision\Images\train_faces\non-face\norm_features"

for dirpath, dnames, fnames in os.walk(img_path):
    for f in fnames:
        full_path = os.path.join(dirpath, f)
        img = cv2.imread(full_path)

        img = scale(img.reshape(img.size,), with_mean=False).reshape(img.shape)

        img = cv2.resize(cv2.imread(full_path, 0), standard_size, interpolation=cv2.INTER_CUBIC)

        fet1 = haar(img, win_size=24, extractor=LEFT_RIGHT, stride=2)
        fet2 = haar(img, win_size=24, extractor=ABOVE_BELOW, stride=2)
        fet3 = haar(img, win_size=24, extractor=DIAGONAL, stride=2)
        fet4 = haar(img, win_size=24, extractor=LINE, stride=2)
        fet = np.stack((fet1, fet2, fet3, fet4)).flatten()

        np.save(fr'{fet_path}\{f[:f.find('.')]}', fet)