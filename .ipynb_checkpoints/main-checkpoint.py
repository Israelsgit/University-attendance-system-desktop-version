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
face_embeddings = np.load('face_embeddings_done_35classes.npz')
Y = face_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascades = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

model = pickle.load(open('svm_model_160x160.pkl', 'rb'))

cap = cv.VideoCapture(0)

#WHILE LOOP
while cap.isOpened():
    ret, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = haarcascades.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in face:
        img = rgb_img[y:y+h, x:x+w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        ypreds = facenet.embeddings(img)
        face_name = model.predict(ypreds)
        final_name = encoder.inverse_transform(face_name)
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 10)
        cv.putText(frame, str(final_name), (x,y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 3, cv.LINE_AA)
        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & ord('q') == 27:
            break

    cap.release()
    cv.destroyAllWindows()