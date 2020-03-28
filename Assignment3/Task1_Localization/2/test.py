
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


ip_folder = "/users/home/dlagroup5/wd/Assignment3/Task1_Localization/Task2/input"
output_folder = "./results"
# X = []
X_test = np.load(ip_folder + "/X_test.npy")
y_test = np.load(ip_folder + "/y_test.npy")
# cv2.imwrite("abc.jpg", X_test[0])

# X_test = X_test[0]
# X.append(X_test)
# X = np.array(X)
# X_test = X


print (X_test.shape, y_test.shape)

R, C = 1572, 1672

inp = Input(shape=(1572,1672,3))
x = Conv2D(64, kernel_size=15, strides=5, activation="relu")(inp)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=4, strides=2)(x)
x = Dropout(rate=0.4)(x)
x = Flatten()(x)

x = Dense(64, activation="linear")(x)
x = Dense(16, activation="linear", name="regression")(x)

model = Model(inputs=inp, outputs=[x])
print (model.output_shape)

model.compile(optimizer="adam",
              loss={"regression": "mean_squared_error"},
              metrics=['accuracy'])

model.summary()
model.load_weights(output_folder + "/model.h5")

print ("Before prediction")
y_pred = model.predict(X_test)
print ("After prediction")

print (y_pred)
img = X_test[2]
for i in range(4):
    rect = cv2.rectangle(img, (y_pred[2][i*4 + 0], y_pred[2][i*4 + 1]), (y_pred[2][i*4 + 2], y_pred[2][i*4 + 3]), (0, 255, 0), 2)
cv2.imwrite("test_bb.jpg", rect)

print ("Saving...")

np.save(output_folder + "/y_pred.npy", y_pred)

np.save(output_folder + "/y_test.npy", y_test)

print ("Saved...")

# print(cm_y1)
