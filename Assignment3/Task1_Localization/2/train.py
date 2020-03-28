#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
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


X_train = np.load(ip_folder + "/X_train.npy")
y_train = np.load(ip_folder + "/y_train.npy")

X_test = np.load(ip_folder + "/X_test.npy")
y_test = np.load(ip_folder + "/y_test.npy")


print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


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
print ("before train")
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=5, batch_size=32)
print ("after train")
model.save_weights("./results/model.h5")
print ("Model saved...")

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig("train_acc.png", bbox_inches='tight')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.savefig("train_loss.png", bbox_inches='tight')
plt.show()
