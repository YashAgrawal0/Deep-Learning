#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import cv2
import os


# In[17]:


image_folder = "/users/home/dlagroup5/wd/Assignment3/data/Q2/Data_test"
mask_folder = "/users/home/dlagroup5/wd/Assignment3/data/Q2/Mask_test"
output_folder = "/users/home/dlagroup5/wd/Assignment3/Task2_Segmentation/data_extract"


# In[18]:


img_names = []
mask_names = []
for file in os.listdir(image_folder):
    img_names.append(image_folder + "/" + file)
    mask_names.append(mask_folder + "/" + "_groundtruth_(1)_" + file.replace("_original",""))
# cv2.imshow("image",cv2.imread(img_names[0]))
# cv2.imshow("image2",cv2.imread(mask_names[0]))
# cv2.waitKey()
# cv2.destroyAllWindows()
noOfImages = len(img_names)
# noOfImages = 10
print(noOfImages)


# In[19]:


def binaryConversion(img):
    arr = np.zeros([img.shape[0],img.shape[1],2])
#2 channel image label
    # 1st Channel - black
    # 2nd Channel - white
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j]>100):
                arr[i][j][1]=1
            else:
                arr[i][j][0]=1
    # print(arr.shape)
    return arr


# In[20]:


def getData(low,high):
    X_data, Y_data = [],[]
    for i in range(low,high):
        img = cv2.imread(img_names[i])
        mask = cv2.imread(mask_names[i],0)
        X_data.append(img)
        # Y_data.append(binaryConversion(mask))
        Y_data.append(mask)
        print(i)
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    # print(X_data.shape,Y_data.shape)
    return X_data,Y_data


# In[21]:


# print("**** TRAIN ****\n")
# X_train, Y_train = getData(0,int(0.8*noOfImages))
print("**** TEST ****\n")
X_test, Y_test = getData(int(0.8*noOfImages),noOfImages)


# In[24]:


# l = Y_train[0][:,:,0]
# for i in range(l.shape[0]):
#     print(l[i])
    


# In[26]:


# x=cv2.imread(mask_names[0],0)
# for i in range(x.shape[0]):
#     print(x[i])


# In[22]:


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# np.save(output_folder + "/x_train.npy", X_train)
# np.save(output_folder + "/y_train.npy", Y_train)
np.save(output_folder + "/x_test.npy", X_test)
np.save(output_folder + "/y_test.npy", Y_test)


# In[ ]:




