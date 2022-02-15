import tensorflow as tf
import cv2
import numpy as np


cap = cv2.VideoCapture('video.mp4')

model = tf.keras.models.load_model('model2')

while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(64,32))
    cv2.imshow('Autoencoder',frame)

    frame  = frame.reshape(-1,32,64,3)
    output = model(frame)
    #print(type(output))
    output = np.array(output)
    output = np.squeeze(output, axis=0)

    # output = cv2.resize(output,(64,32))
    #print(output.shape)
    cv2.imshow("Prediction",output)
    k=cv2.waitKey(1)
    if(k==27):
        cv2.destroyAllWindows()
        break