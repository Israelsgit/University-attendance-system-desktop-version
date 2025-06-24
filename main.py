# FACE RECOGNITION PART II
#IMPORTS
import cv2 as cv
from cv2.data import haarcascades
from keras_facenet import FaceNet
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#INITIALIZE
facenet = FaceNet()
faces_embeddings = np.load('faces_embeddings_done_35classes.npz')
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
model = pickle.load(open('svm_model_160x160.pkl', 'rb'))

cap = cv.VideoCapture(0)

#WHILE LOOP
while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for (x, y, w, h) in faces:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160)) #1x160x160x3
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        face_name = model.predict(ypred)
        final_name = encoder.inverse_transform(face_name)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 3, cv.LINE_AA)

    cv.imshow("Face Recognition: ", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows()