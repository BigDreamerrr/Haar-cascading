import os
import random

def perserve_60_percent(path):
    files = os.listdir(path)

    for f in files:
        num = random.uniform(0, 1)
        if num > 0.6:
            os.remove(f'{path}\\{f}')

perserve_60_percent(r"D:\Computer vision\Images\train_faces\non-face\images\archive (2)\afhq\train\cat")
perserve_60_percent(r"D:\Computer vision\Images\train_faces\non-face\images\archive (2)\afhq\train\dog")
perserve_60_percent(r"D:\Computer vision\Images\train_faces\non-face\images\archive (2)\afhq\train\wild")