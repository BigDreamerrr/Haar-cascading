import cv2
from c_adaboost import integral
from sklearn.preprocessing import scale
import numpy as np
import math

def sum(integral, x1, y1, x2, y2):
    C = integral[x2][y2]
    A = 0
    B = 0
    D = 0

    if x1 >= 1 and y1 >= 1:
        A = integral[x1 - 1][y1 - 1]
    if x1 >= 1:
        B = integral[x1 - 1][y2]
    if y1 >= 1:
        D = integral[x2][y1 - 1]

    return C + A - B - D

def light_invariant(integral_img1, integral_img2, start, end, sub_img):
    total_in_non_squared = sum(integral_img1, start[0], start[1], end[0], end[1])
    total_in_squared = sum(integral_img2, start[0], start[1], end[0], end[1])

    return sub_img / math.sqrt(
        (total_in_squared / sub_img.size - (total_in_non_squared / sub_img.size)**2))

def evaluate(num_tests=100):
    for _ in range(num_tests):

        img = cv2.imread(fr"D:\test{np.random.randint(0, 5)}.jpg", 0).astype(np.float64)
        integral_img1 = integral(img)
        integral_img2 = integral(img**2)

        start1 = np.random.randint(0, img.shape[0] - 40)
        start2 = np.random.randint(0, img.shape[1] - 40)
        
        win_size = np.random.randint(10, 40)

        cropped = img[start1:start1 + win_size, start2:start2 + win_size]
        scaled_img = scale(cropped.flatten(), with_mean=False).reshape(cropped.shape)

        # var(X) = E(X^2) - E(X)^2
        my_scaled_img = light_invariant(
            integral_img1, 
            integral_img2, 
            (start1, start2), 
            (start1 + win_size - 1, start2 + win_size - 1), 
            cropped)


        if not np.allclose(my_scaled_img, scaled_img):
            print('wrong!')

evaluate(num_tests=5_000)