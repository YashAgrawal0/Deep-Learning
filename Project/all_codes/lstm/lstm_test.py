#!/usr/bin/env python
# coding: utf-8

# In[85]:

import pickle
import numpy as np
import math
import matplotlib
import os
import cv2
import random
from sklearn.utils import shuffle
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import pandas
import numpy as np
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Flatten
from keras.layers import LSTM, Dropout, ConvLSTM2D, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


THIS_PATH = "/users/home/dlagroup5/wd/Project/final_annotation"
INPUT_VIDEO = THIS_PATH + "/input/video.mp4"
model_weights_path = THIS_PATH + "/users/home/dlagroup5/wd/Project/code/lstm/results/model_mse.h5"
op_folder = THIS_PATH + "/results"
FRAME_RATE = 0.3
look_back = 40
ip_shape = [197, look_back, 72, 128, 3]
output_layer_size = 84
scaling = 0.1

error_file = open(THIS_PATH + "/error.txt", "w")
error_file.write("")
error_file.close()
error_file = open(THIS_PATH + "/error.txt", "a")

def fill(times, img):
    arr = []
    for i in range(0, times):
        arr.append(img)
    return np.array(arr)

def getFrameUtil(sec, vidcap, frame_no):
	vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	hasFrames,image = vidcap.read()
	image_small = 0
	if(hasFrames):
		image_small = cv2.resize(image, (0,0), fx=scaling, fy=scaling)
	return hasFrames, image, image_small

def getFrame(vidcap):
	success = True
	frame_no, sec = 0, -FRAME_RATE
	frames, frames_small = [], []
	while success:
		# if(sec % 10 == 0):
		# 	print(sec)
		sec = sec + FRAME_RATE
		sec = round(sec, 2)
		success, img, img_small = getFrameUtil(sec, vidcap, frame_no)
		frame_no += 1
		if(success == True):
			frames.append(img)
			frames_small.append(img_small)
	frames = np.array(frames)
	frames_small = np.array(frames_small)
	return frames, frames_small

vidcap = cv2.VideoCapture(INPUT_VIDEO)
frames, frames_small = getFrame(vidcap)
print(frames.shape, frames_small.shape)

error_file.write("Converted to frames\n")

total_img = frames_small.shape[0]
curr_img = np.ndarray.tolist(frames_small)

if(total_img < look_back):
	left = (look_back - total_img) // 2
	right = look_back - total_img - left
	padL = fill(left, curr_img[0])
	padR = fill(right, curr_img[total_img - 1])
	img_set = np.array(curr_img)
	# print(padL.shape, img_set.shape, padR.shape)
	paddedL = np.concatenate((padL, img_set), axis=0)
	paddedAll = np.concatenate((paddedL, padR), axis=0)
	curr_img = np.array(paddedAll)

elif(total_img > look_back):
	left = (total_img - look_back) // 2
	right = total_img - (total_img - look_back - left)
	curr_img = curr_img[left:right]
	curr_img = np.array(curr_img)

else: curr_img = np.array(curr_img)

curr_img = np.expand_dims(curr_img, axis=0)

print("Padded: ", curr_img.shape)

error_file.write("Model testing started\n")

model = Sequential()
model.add(ConvLSTM2D(filters=32, kernel_size=8, strides=5, data_format="channels_last",
	input_shape=(ip_shape[1], ip_shape[2], ip_shape[3], ip_shape[4]), return_sequences=True))
# model.add(AveragePooling3D(pool_size=(3, 3, 3)))
model.add(Flatten())
# model.add(BatchNormalization())
# model.add(LSTM(256, input_shape=(look_back, 67456)))
# model.add(Dropout(0.4))
# model.add(Dense(512, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(output_layer_size, activation="relu"))

optimizer = RMSprop(lr=0.06)
model.compile(loss='mse', optimizer="adam", metrics=["accuracy"])
model.summary()

error_file.write("model weights loading!\n")
model.load_weights(model_weights_path)
error_file.write("model weights loaded!\n")

print("Weights loaded")

Y_pred = model.predict(curr_img)

print(Y_pred)
np.save(op_folder + "/lstm_pred.npy", Y_pred)
print("output saved to: " + str(op_folder))

error_file.write("DONE!\n")
