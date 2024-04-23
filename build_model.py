import os
os.add_dll_directory(r"D:\Computer vision\Adaboost")

import numpy as np
from c_adaboost import SimpleAdaBoost
from c_adaboost import CONTINOUS
from sklearn.model_selection import train_test_split

# path = r"D:\Computer vision\Images\train_faces\norm_fet"
# none_face_path = r"D:\Computer vision\Images\train_faces\non-face\norm_features"

# fet_size = 15984

# files = os.listdir(path)
# none_face_files = os.listdir(none_face_path)

# X = np.empty((len(files) + len(none_face_files), fet_size))
# Y = np.empty((len(files) + len(none_face_files)))

# for index, f in enumerate(files):
#     fet = np.load(fr"{path}\{f}")
#     X[index] = fet
#     Y[index] = 1

# offset = len(files)

# for index, f in enumerate(none_face_files):
#     fet = np.load(fr"{none_face_path}\{f}")
#     X[index +offset] = fet
#     Y[index + offset] = -1

# types = [CONTINOUS] * len(X[0])

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.33, random_state=42, stratify=Y)

# np.save('x_train', X_train)
# np.save('y_train', y_train)

# np.save('x_test', X_test)
# np.save('y_test', y_test)

X_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

X_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

types = [CONTINOUS] * len(X_train[0])

model = SimpleAdaBoost(types, 40, 150)
print('training...')
model.fit(X_train, y_train)
model.save(r'D:', 'norm_cascade3')

print('predict...')
pred = model.predict(X_test)

count = 0
for index, data in enumerate(y_test):
    if data * pred[index] > 0:
        count += 1

print(count)
print(len(y_test))
print(count / len(y_test))