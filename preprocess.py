import cv2
import os
import numpy as np

def get_crops(path):
    face_detector = cv2.dnn.readNetFromCaffe(
        r"params/deploy.prototxt.txt", 
        r"params/res10_300x300_ssd_iter_140000.caffemodel")

    image = cv2.imread(path)

    # Get the height and width of the input image
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Feed the blob as input to the DNN Face Detector model
    face_detector.setInput(blob)
    detections = face_detector.forward()

    crop = None
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        startX = max(0, startX)
        startY = max(0, startY)

        endX = min(endX, image.shape[1] - 1)
        endY = min(endY, image.shape[0] - 1)

        # Filter out weak detections
        if confidence > 0.5 and (endX - startX) >= 50 and (endY - startY) >= 50:
            if crop is not None:
                return None
            
            crop = image[startY:endY, startX:endX]

    return crop

import pathlib

def extract_data(file_name):
    underscore_pos = file_name.index('_')
    age = int(file_name[:underscore_pos])
    file_name = file_name[underscore_pos+1:]
    
    underscore_pos = file_name.index('_')
    gender = int(file_name[:underscore_pos])

    file_name = file_name[underscore_pos+1:]
    try:
        return age_group(age), gender, int(file_name[:file_name.index('_')])
    except:
        return None, None, None

def age_group(age):
    if age <= 5:
        return 0
    elif age <= 22:
        return 1
    elif age <= 50:
        return 2
    elif age <= 65:
        return 3
    
    return 4

cnts = {}

def crop_and_save(paths):
    img_index = 0

    stats = {}
    for path in paths:
        for dirpath, _, fnames in os.walk(path):
            for f in fnames:
                if (not f.endswith('.jpg')) and (not f.endswith('.png')):
                    continue

                full_path = os.path.join(dirpath, f)
                crop = get_crops(full_path)

                age, gender, race = extract_data(f)

                if age == None:
                    continue
                
                if crop is not None and cnts.get((age, gender, race), 0) < 400:
                    cnts[(age, gender, race)] = cnts.get((age, gender, race), 0) + 1

                    cv2.imwrite(
                        fr'D:\Computer vision\Images\train_faces\processed2\{age}_{img_index}{pathlib.Path(f).suffix}', crop)
                    img_index += 1
                    stats[age] = stats.get(age, 0) + 1

    np.save('stats.npy', stats)

crop_and_save([
    r"D:\Computer vision\Images\train_faces\part1",
    r"D:\Computer vision\Images\train_faces\part2",
    r"D:\Computer vision\Images\train_faces\part3"
])