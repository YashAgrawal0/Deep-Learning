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


ip_folder = "/users/home/dlagroup5/wd/Assignment3/Task1_Localization/Task1/input"


X_train = np.load(ip_folder + "/X_train.npy")
y1_train = np.load(ip_folder + "/y1_train.npy")
y2_train = np.load(ip_folder + "/y2_train.npy")

X_test = np.load(ip_folder + "/X_test.npy")
y1_test = np.load(ip_folder + "/y1_test.npy")
y2_test = np.load(ip_folder + "/y2_test.npy")


print (X_train.shape, y1_train.shape, y2_train.shape)
print (X_test.shape, y1_test.shape, y2_test.shape)


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
print ("before train")
history = model.fit(X_train, [y1_train, y2_train],
                    validation_data=(X_test, [y1_test, y2_test]),
                    epochs=3, batch_size=300)
print ("after train")
model.save_weights("model.h5")
print ("Model saved...")

# Plot training & validation accuracy values
plt.plot(history.history['class_head_acc'])
plt.plot(history.history['reg_head_acc'])
plt.plot(history.history['val_class_head_acc'])
plt.plot(history.history['val_reg_head_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Classification Train',
            'Regression Train',
            'Classification Test',
            'Regression Test'], loc='lower right')
plt.savefig("train_acc.png", bbox_inches='tight')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['class_head_loss'])
plt.plot(history.history['reg_head_loss'])
plt.plot(history.history['val_class_head_loss'])
plt.plot(history.history['val_reg_head_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss'], loc='upper right')
plt.savefig("train_loss.png", bbox_inches='tight')
plt.show()

y1_pred, y2_pred = model.predict(X_test)

y1_test = y1_test >= 0.5
y2_test = y2_test >= 0.5

y1_pred = y1_pred >= 0.5
y2_pred = y2_pred >= 0.5

cm_y1 = confusion_matrix(y1_test.argmax(axis=1), y1_pred.argmax(axis=1))
cm_y2 = confusion_matrix(y2_test.argmax(axis=1), y2_pred.argmax(axis=1))

print ("Conf matrix for classification:")
print(cm_y1)
print ("Conf matrix for regression:")
print(cm_y2)
