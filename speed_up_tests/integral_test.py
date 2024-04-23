from c_adaboost import integral
from c_adaboost import haar

import cv2
import numpy as np

img = cv2.imread(r"D:\test.jpg", 0)
integral_img = integral(img)

sum = 0
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        sum += img[x][y]

sum2 = np.sum(img)
if sum != integral_img[-1][-1]:
    print('Wrong!')