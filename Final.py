import cv2
import numpy as np
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


dataset = cv2.CascadeClassifier(r"data.xml")

names = { 0 : 'Mask', 1 : 'No Mask'}
x_train = np.load(r"maskedUnmasked\xTrain.npy")
x_test = np.load(r"maskedUnmasked\xTest.npy")
y_train = np.load(r"maskedUnmasked\yTrain.npy")
y_test = np.load(r"maskedUnmasked\yTest.npy")

x_train = x_train.reshape(5823, 50*50*3)
x_test = x_test.reshape(597, 50*50*3)

pca = PCA(n_components = 3)

x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)
print(x_test.shape, x_train.shape)

model = SVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test, y_pred))




vid  = cv2.VideoCapture(0)

data = []

while True:
    ret, frame = vid.read()
    if ret:
        faces = dataset.detectMultiScale(frame, 1.2)
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 4 )
            face = frame[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50, 50))
            face = face.reshape(1, -1)
            face = pca.transform(face)
            pred = model.predict(face)[0]
            n = names[int(pred)]
            print(n)

        cv2.imshow("result", frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Camera Not Found")

vid.release()
cv2.destroyAllWindows()