import os
os.add_dll_directory(r"D:\Computer vision\Adaboost4")

import numpy as np
from c_adaboost import SimpleAdaBoost
from c_adaboost import CONTINOUS

X_train = np.load('x_train.npy')
X_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

types = [CONTINOUS] * len(X_train[0])

model = SimpleAdaBoost(types, 40, 1_000)
model.load(r'D:\norm_cascade2')

print('training...')
model.fit(X_train, y_train)
model.save(r'D:', 'norm_cascade2')

print('predict...')
pred = model.predict(X_test)

count = 0
for index, data in enumerate(y_test):
    if data * pred[index] > 0:
        count += 1

print(count)
print(len(y_test))
print(count / len(y_test))