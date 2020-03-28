import numpy as np
import os
import cv2
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras.models import model_from_json
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


phase = str(sys.argv[1])

def createData(dirPath):
  lst = os.listdir(dirPath)
  lst.sort()

  data = []
  ratio = []

  for i in range(len(lst)):
    img = cv2.imread(dirPath+'/'+lst[i])
    line = []
    x = 300/img.shape[0]
    y = 300/img.shape[1]
    line.append(x)
    line.append(y)
    ratio.append(line)
    img = cv2.resize(img, dsize=(300,300  ), interpolation=cv2.INTER_CUBIC)
    data.append(img)
    print("Input Image :",i)
  
  return np.array(data),np.array(ratio),lst

def createGroundTruth(dirPath, ratio):
  lst = os.listdir(dirPath)
  lst.sort()

  gt = []

  for i in range(len(lst)):
    x,y = open(dirPath+'/'+lst[i],"r").readline().split()
    line = []
    line.append(int(x))
    line.append(int(y))
    gt.append(line)

  return np.multiply(ratio, np.array(gt))


if phase=='train':

  EPOCHS = int(sys.argv[2])
  dataFolder = str(input("Enter the name of training folder having 'Data' and 'Ground_truth' as folders :  "))
  data_path = dataFolder + '/Data'
  gtPath = dataFolder+'/Ground_truth'
  train_data, train_data_ratio,lst = createData(data_path)
  gt = createGroundTruth(gtPath, train_data_ratio)
  
  model = Sequential()
  model.add(Conv2D(32, kernel_size=10, strides=5, activation="relu", input_shape=(300,300,3)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=2, strides=2))
  model.add(Flatten())
  model.add(Dense(2, activation='linear'))
  model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

  history = model.fit(train_data, gt,epochs=EPOCHS)
  
  model.save_weights('model.h5')


if phase=='test':

  print("Provide images in JPEG format")

  dataFolder = str(input("Enter the name of testing folder having folder named as 'Data'  :  "))
  testdata_path = dataFolder + '/Data'
  test_data, test_data_ratio, lst = createData(testdata_path)

  model = Sequential()
  model.add(Conv2D(32, kernel_size=10, strides=5, activation="relu", input_shape=(300,300,3)))
  model.add(BatchNormalization())
  model.add(MaxPooling2D(pool_size=2, strides=2))
  model.add(Flatten())
  model.add(Dense(2, activation='linear'))
  model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
  
  model.load_weights('model.h5')

  op = model.predict(test_data)
  finalOutput = np.divide(op,test_data_ratio)
  
  dirName = 'outputGroundTruths'
  os.mkdir(dirName)

  print(finalOutput.shape)
  for i in range(len(finalOutput)):
    x,y = finalOutput[i]
    fileName = lst[i]
    fileName = fileName[:-5]
    f = open(dirName+"/"+fileName+".txt","w+")
    f.write(str(x)+" "+str(y))
    f.close()


  # for i in range(len(test_data)):
  #   x,y = finalOutput[i] 

  #   fileName = lst[i]
  #   fileName = fileName[:-4]

  #   f = open(dirName+"/"+fileName+"txt","w+")
  #   f.write(str(int(x))+" "+str(int(y))
  #   # f.close()