from c_adaboost import integral
from c_adaboost import LEFT_RIGHT, ABOVE_BELOW, DIAGONAL, LINE
from c_adaboost import haar
from c_adaboost import FastHaarFetGetter

import cv2
import numpy as np

def evaluate(num_tests=100):
    # win_size, extractor(0 - 3), stride
    for _ in range(num_tests):
        win_size = np.random.randint(15, 40)
        extractor = np.random.randint(0, 4)
        stride = np.random.randint(1, 10)

        img = cv2.imread(fr"D:\test{np.random.randint(0, 5)}.jpg", 0)

        fet_vec = haar(img, win_size=win_size, extractor=extractor, stride=stride)

        haar_getter = FastHaarFetGetter(img)

        for i in range(fet_vec.shape[0]):
            for j in range(fet_vec.shape[1]):
                fet = haar_getter.get(win_size, extractor, stride, i, j)

                if fet != fet_vec[i][j]:
                    print('wrong!')

evaluate()