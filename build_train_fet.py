import os
os.add_dll_directory(r"D:\Computer vision\Adaboost")

import cv2
import numpy as np
import cv2
from c_adaboost import haar
from c_adaboost import LEFT_RIGHT, ABOVE_BELOW, DIAGONAL, LINE
from sklearn.preprocessing import scale

standard_size= (131, 170)

path = r'D:\Computer vision\Images\train_faces\processed2'
files = os.listdir(path)

close = True

for f in files:
    img = cv2.imread(f"{path}\\{f}")
    lower_bound = min(img.shape[0], img.shape[1])

    img = cv2.cvtColor(cv2.resize(img, standard_size, interpolation=cv2.INTER_CUBIC), 
                       cv2.COLOR_BGR2GRAY)
    
    img = scale(img.reshape(img.size,), with_mean=False).reshape(img.shape)

    fet1 = haar(img, win_size=24, extractor=LEFT_RIGHT, stride=2)
    fet2 = haar(img, win_size=24, extractor=ABOVE_BELOW, stride=2)
    fet3 = haar(img, win_size=24, extractor=DIAGONAL, stride=2)
    fet4 = haar(img, win_size=24, extractor=LINE, stride=2)
    fet = np.stack((fet1, fet2, fet3, fet4)).flatten()

    f = f[:f.find('.')]
    np.save(fr'D:\Computer vision\Images\train_faces\norm_fet\{f}.npy', fet)