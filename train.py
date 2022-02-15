import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import cv2
from tensorflow.keras.datasets import mnist
import glob
###########################

def load_data():
    x_train = []
    y_train = []
    print("Processing...")
    for file in glob.glob("train\\input\\*jpg"):
        img = cv2.imread(file)
        img = cv2.resize(img,(64,32))
        x_train.append(img)

    for file in glob.glob("train\\output\\*jpg"):
        img = cv2.imread(file,0)
        img = cv2.resize(img,(64,32))
        y_train.append(img)
    print("\nDone\n")

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    print("Input Shape :" + str(x_train.shape))
    print("Output Shape :" + str(y_train.shape))


    x_val = []
    y_val = []
    print("Processing...")
    for file in glob.glob("train\\input_val\\*jpg"):
        x_val.append(cv2.imread(file))

    for file in glob.glob("train\\output_val\\*jpg"):
        # img = cv2.imread(file)
        # img = img.reshape()
        y_val.append(cv2.imread(file,0))
    print("\nDone\n")

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    print("Input Shape :" + str(x_val.shape))
    print("Output Shape :" + str(y_val.shape))


    x_train = x_train/255
    y_train = y_train/255
    x_val = x_val/255
    y_val = y_val/255


    return x_train,y_train,x_val,y_val

##############################################
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)

# x_train = x_train/255.0

x_train,y_train,x_val,y_val = load_data()
#Encoder
x = Input(shape = (32,64,3))
c1 = Conv2D(32,3,activation="relu",padding="same")(x)
# c1 = Conv2D(32,3,activation="relu",padding="same")()
# b1 = MaxPool2D((2,2))(c1)
c2 = Conv2D(64,3,activation="relu",padding="same")(c1)
c3 = Conv2D(64,3,activation="relu",padding="same")(c2)
f1 = Flatten()(c3)
print("F1 Shape : "+str(f1.shape))
d1 = Dense(100,activation="relu")(f1)
d2 = Dense(50,activation="relu")(d1)
# print("C3 shape" + str(c3.shape))
# #Decoder
d3 = Dense(100,activation="relu")(d2)
d4 = Dense(int(f1.shape[0]),activation="relu")(d3)
R = Reshape((32,64,64))(d4)
up1 = Conv2DTranspose(64,3,activation="relu",padding="same")(R)
up2 = Conv2DTranspose(64,3,activation="relu",padding="same")(up1)
up3 = Conv2DTranspose(32,3,activation="relu",padding="same")(up2)
up4 = Conv2DTranspose(1,3,activation="relu",padding="same")(up3)

model = tf.keras.models.Model(x,up4,name="ae")
model.summary()



# class MyCustomCallback(tf.keras.callbacks.Callback):
#     def printoutput(self):

# print(up3.shape)

model.compile(optimizer='Adam',loss = tf.keras.losses.MSE,metrics=["mean_squared_error"])

with tf.device('GPU'):
    model.fit(x_train,y_train,epochs=100,batch_size=10)

model.save('model4')

