import os
import cv2
import numpy as np

dataset = cv2.CascadeClassifier(r"C:\Users\hp\Desktop\New\data.xml")

x_train =[]
x1_train = []
x2_train = []
x_test = []
y_train = []
y_test = []
x1_cnt = 0
x2_cnt = 0

for filename in os.listdir(r"Dataset\Face Mask Dataset\Train\WithMask"):
    img = cv2.imread(os.path.join(r"Dataset\Face Mask Dataset\Train\WithMask", filename))
    if img is not None:
        faces = dataset.detectMultiScale(img, 1.2)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 4 )
            #face = img[y:y+h, x:x+w, :]
            face = img
            face = cv2.resize(face, (50, 50))
            x1_cnt += 1
            x1_train.append(face)

for filename in os.listdir(r"Dataset\Face Mask Dataset\Train\WithoutMask"):
    img = cv2.imread(os.path.join(r"Dataset\Face Mask Dataset\Train\WithoutMask", filename))
    if img is not None:
        faces = dataset.detectMultiScale(img, 1.2)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 4 )
            #face = img[y:y+h, x:x+w, :]
            face = img
            face = cv2.resize(face, (50, 50))
            x2_cnt += 1
            x2_train.append(face)

x_train = np.concatenate((x1_train, x2_train))

y_train = np.zeros(x1_cnt + x2_cnt)
y_train[x1_cnt:] = 1
print(x1_cnt + x2_cnt)
np.save(r"maskedUnmasked\xTrain.npy", x_train)
np.save(r"maskedUnmasked\yTrain.npy", y_train)

x_train =[]
x1_train = []
x2_train = []
x_test = []
y_train = []
y_test = []
x1_cnt = 0
x2_cnt = 0

for filename in os.listdir(r"Dataset\Face Mask Dataset\Test\WithMask"):
    img = cv2.imread(os.path.join(r"Dataset\Face Mask Dataset\Test\WithMask", filename))
    if img is not None:
        faces = dataset.detectMultiScale(img, 1.2)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 4 )
            #face = img[y:y+h, x:x+w, :]
            face = img
            face = cv2.resize(face, (50, 50))
            x1_cnt += 1
            x1_train.append(face)

for filename in os.listdir(r"Dataset\Face Mask Dataset\Test\WithoutMask"):
    img = cv2.imread(os.path.join(r"Dataset\Face Mask Dataset\Test\WithoutMask", filename))
    if img is not None:
        faces = dataset.detectMultiScale(img, 1.2)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 4 )
            #face = img[y:y+h, x:x+w, :]
            face = img
            face = cv2.resize(face, (50, 50))
            x2_cnt += 1
            x2_train.append(face)

x_test = np.concatenate((x1_train, x2_train))

y_test = np.zeros(x1_cnt + x2_cnt)
y_test[x1_cnt:] = 1
print(x1_cnt + x2_cnt)
np.save(r"maskedUnmasked\xTest.npy", x_test)
np.save(r"maskedUnmasked\yTest.npy", y_test)