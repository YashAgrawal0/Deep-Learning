#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import cv2
import os


# In[50]:


image_folder = "/users/home/dlagroup5/wd/Assignment3/data/Q2/Data"
mask_folder = "/users/home/dlagroup5/wd/Assignment3/data/Q2/Mask"
output_folder = "/users/home/dlagroup5/wd/Assignment3/Task2_Segmentation/data_extract"


# In[51]:


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
print(noOfImages)


# In[52]:


def getData(low,high):
    X_data, Y_data = [],[]
    for i in range(low,high):
        img = cv2.imread(img_names[i])
        mask = cv2.imread(mask_names[i])
        X_data.append(img)
        Y_data.append(mask)
        print(i,img.shape,mask.shape)
    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    return X_data,Y_data


# In[53]:


print("**** TRAIN ****\n")
X_train, Y_train = getData(0,int(0.8*noOfImages))
print("**** TEST ****\n")
X_test, Y_test = getData(int(0.8*noOfImages),noOfImages)


# In[48]:


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
np.save(output_folder + "/x_train.npy", X_train)
np.save(output_folder + "/y_train.npy", Y_train)
np.save(output_folder + "/x_test.npy", X_test)
np.save(output_folder + "/y_test.npy", Y_test)


# In[ ]:




