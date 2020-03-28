
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D, Dropout, Input
from keras import metrics
from keras.models import model_from_json
from keras.callbacks import Callback
from keras import optimizers
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


ip_folder = "/users/home/dlagroup5/wd/Assignment3/Task1_Localization/Task1/input"
output_folder = "results"

X_test = np.load(ip_folder + "/X_test.npy")
y1_test = np.load(ip_folder + "/y1_test.npy")
y2_test = np.load(ip_folder + "/y2_test.npy")


print (X_test.shape, y1_test.shape, y2_test.shape)

R, C = 480, 640

def getBranch(inp, out_neuron, out_act, name, parity):
    if(parity == 0):
        x = Dense(64, activation="relu")(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
    else:
        x = Dense(64, activation="linear")(inp)
    x = Dense(out_neuron, activation=out_act, name=name)(x)
    return x


inp = Input(shape=(480,640,3))
cnn = Conv2D(64, kernel_size=10, strides=5, activation="relu")(inp)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D(pool_size=3, strides=2)(cnn)
cnn = Dropout(rate=0.4)(cnn)
cnn = Flatten()(cnn)


classification_head = getBranch(cnn, 3, "softmax", "class_head", 0)
regression_head = getBranch(cnn, 4, "linear", "reg_head", 1)
model = Model(inputs=inp, outputs=[classification_head, regression_head])
print (model.output_shape)

model.compile(optimizer="adam",
              loss={"class_head": "categorical_crossentropy", "reg_head": "mean_squared_error"},
              metrics=['accuracy'])

model.summary()
model.load_weights(output_folder + "/model.h5")

print ("Before prediction")
y1_pred, y2_pred = model.predict(X_test)
print ("After prediction")

# print (y1_pred.tolist())
# print (y2_pred.tolist())


y1_test = y1_test >= 0.5

y1_pred = y1_pred >= 0.5

cm_y1 = confusion_matrix(y1_test.argmax(axis=1), y1_pred.argmax(axis=1))

print ("Saving...")

np.save(output_folder + "/y1_pred.npy", y1_pred)
np.save(output_folder + "/y2_pred.npy", y2_pred)

np.save(output_folder + "/y1_test.npy", y1_test)
np.save(output_folder + "/y2_test.npy", y2_test)

print ("Saved...")

print(cm_y1)
