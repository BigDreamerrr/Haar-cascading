from c_adaboost import SimpleAdaBoost
from c_adaboost import haar
import numpy as np
from c_adaboost import LEFT_RIGHT, ABOVE_BELOW, DIAGONAL, LINE
from c_adaboost import integral, FastHaarFetGetter
import ctypes
import cv2
from sklearn.preprocessing import scale
import math
import time

block_size = 3996
original_fet_shape =(74, 54)

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

    contrast_ratio = math.sqrt(
        (total_in_squared / sub_img.size - (total_in_non_squared / sub_img.size)**2))

    if contrast_ratio == 0:
        return sub_img

    return sub_img / contrast_ratio

def remover(cords, i, j, win_size, vote):
    weak_overlaps = []
    overlap_cnt = 0

    for cord in cords:
        if i >= cord[0] - win_size[0] / 1.5 \
            and i <= cord[0] + win_size[0] / 1.5\
            and j >= cord[1] - win_size[1] / 1.5 \
            and j <= cord[1] + win_size[1] / 1.5:
            
            overlap_cnt += 1 # overlap!
            if vote > cords[cord]:
                weak_overlaps.append(cord)

    if overlap_cnt == len(weak_overlaps): # all overlaps are weak then add this to dict
        for weak in weak_overlaps:
            cords.pop(weak) # pop all weaks
        
        cords[(i, j)] = vote

def scan(model, img, windows, stride=3):
    cords = {}

    integeral1 = integral(img)
    integral2 = integral(img.astype(np.float64)**2)

    indexes = list(model.get_indexes())
    indexes = (ctypes.c_int * len(indexes))(*indexes)
    full_fets = np.array([[0.0] * 15_984])

    cnt_subs = 0
    for window in windows:
        i = 0

        while i < img.shape[0] - window[0]:
            j = 0
            while j < img.shape[1] - window[1]:
                cnt_subs += 1

                # if cnt_subs % 100 == 0:
                #     print(f'{cnt_subs} subwindows slided!')

                invariant = light_invariant(
                    integeral1, 
                    integral2,
                    (i, j), 
                    (i + window[0] - 1, j + window[1] - 1), 
                    img[i:i + window[0], j:j+window[1]].astype(np.float64))

                resized = cv2.resize(invariant, (131, 170), cv2.INTER_CUBIC)
                
                # resized = scale(resized.flatten(), with_mean=False).reshape(resized.shape)
                
                # fet1 = haar(resized, win_size=24, extractor=LEFT_RIGHT, stride=2)
                # fet2 = haar(resized, win_size=24, extractor=ABOVE_BELOW, stride=2)
                # fet3 = haar(resized, win_size=24, extractor=DIAGONAL, stride=2)
                # fet4 = haar(resized, win_size=24, extractor=LINE, stride=2)
                # fet = np.stack((fet1, fet2, fet3, fet4)).flatten()

                haar_getter = FastHaarFetGetter(resized)
                haar_getter.fill(24, 2, block_size, original_fet_shape[1], indexes, 
                                 full_fets)

                # for index in indexes:
                #     extractor_id = index // block_size
                #     offset_in_block = index % block_size
                #     row = offset_in_block // original_fet_shape[1]
                #     col = offset_in_block % original_fet_shape[1]

                #     # full_fets[0][index] = haar_getter.get(24, extractor_id, 2, row, col)
                #     pass

                # votes = model.predict(np.array([fet]))[0]
                votes = model.predict(full_fets)[0]

                if votes > 6.0: # if this vote is high enough then try to fit it to image
                    remover(cords, i, j, window, votes)

                j += stride
            i += stride
            
    return cords.keys()

# "D:\Computer vision\Images\test_faces\train\image_data\10007.jpg"
# 14
# "D:\Computer vision\Images\test_faces\train\image_data\10005.jpg"
# 6.0

# r"D:\Computer vision\Images\test_faces\train\image_data\10115.jpg"
# r"D:\Computer vision\Images\test_faces\train\image_data\10007.jpg"(4.0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def adjusted_detect_face(img):
    face_img = img.copy()
    
    start = time.time()
    face_rect = face_cascade.detectMultiScale(face_img)
    print(time.time() - start)

    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), 
                      (x + w, y + h), (255, 255, 255), 1)\
         
    return face_img
# "D:\Computer vision\Images\test_faces\train\image_data\10167.jpg"
# "D:\Computer vision\Images\test_faces\train\image_data\10124.jpg"
# "D:\Computer vision\Images\test_faces\train\image_data\10195.jpg"windows = [(120, 120)]
# r"D:\Computer vision\Images\test_faces\train\image_data\10114.jpg"windows = [(25, 25)]
# r"D:\Computer vision\Images\test_faces\train\image_data\10116.jpg"
# r"D:\Computer vision\Images\test_faces\train\image_data\10153.jpg"
# "C:\Users\ducg5\OneDrive\Pictures\Camera Roll\WIN_20240417_14_34_17_Pro.jpg"
#r"D:\Computer vision\Images\test_faces\train\image_data\10124.jpg"
# "D:\Computer vision\Images\test_faces\train\image_data\10185.jpg"
# r"D:\Computer vision\Images\test_faces\train\image_data\10221.jpg"

img = cv2.imread(r"D:\Computer vision\Images\test_faces\train\image_data\10185.jpg", 0)
img = cv2.resize(img, (300, 300))

# cv2.imshow('img', img)
# cv2.waitKey(0)

# faster_img = adjusted_detect_face(img)

# cv2.imshow('image', faster_img)
# cv2.waitKey(0)

# (80, 80), (100, 100), (140, 140)
# windows = [(80, 80), (130, 130), (200, 200)]
# windows = [(100, 100)]
windows = [ (40, 40), (80, 80), (120, 120), (160, 160)]

model = SimpleAdaBoost([0])
model.load(r'D:\norm_cascade')

import time

start = time.time()
all_cords = scan(model, img, windows)
print(time.time() - start)

for cord in all_cords:
    cv2.rectangle(img, cord[::-1], (cord[1] + windows[0][1], cord[0] + windows[0][0]),
                   (0, 0, 255), 1)
    
cv2.imshow('test', img)
cv2.waitKey(0)