#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[8]:


# input_folder = "content/My Drive/Sem6/DL/A3/data_extract_new"
# input_folder = "./data_extract_new"
# X_train = np.load(input_folder + "/x_train.npy")
# Y_train = np.load(input_folder + "/y_train.npy")
# X_test = np.load(input_folder + "/x_test.npy")
# Y_test = np.load(input_folder + "/y_test.npy")
X_test = cv2.imread("a.tiff")
Y_test = cv2.imread("b.tiff",0)
img_shape = X_test.shape
# mask_shape = Y_train[0].shape
# print(img_shape,mask_shape)
# Y_train = np.reshape(Y_train,(Y_train.shape[0],Y_train.shape[1]*Y_train.shape[2],2))
# Y_test = np.reshape(Y_test,(Y_test.shape[0],Y_test.shape[1]*Y_test.shape[2],2))


# In[9]:


# def create_encoding_layers():
#     kernel = 3
#     filter_size = 64
#     pad = 1
#     pool_size = 2
#     stride = 1
#     return [
#         ZeroPadding2D(padding=(pad,pad)),
#         Conv2D(filter_size, kernel_size=kernel, strides=1, padding='valid'),
#         BatchNormalization(),
#         Activation('relu'),
#         MaxPooling2D(pool_size=(pool_size, pool_size)),

#         ZeroPadding2D(padding=(pad,pad)),
#         Conv2D(128, kernel_size=kernel, strides=1, padding='valid'),
#         BatchNormalization(),
#         Activation('relu'),
# #         MaxPooling2D(pool_size=(pool_size, pool_size)),

# #         ZeroPadding2D(padding=(pad,pad)),
# #         Conv2D(256, kernel_size=kernel, strides=1, padding='valid'),
# #         BatchNormalization(),
# #         Activation('relu'),
# #         MaxPooling2D(pool_size=(pool_size, pool_size)),

# #         ZeroPadding2D(padding=(pad,pad)),
# #         Conv2D(512, kernel_size=kernel, strides=1, padding='valid'),
# #         BatchNormalization(),
# #         Activation('relu'),
#     ]

# def create_decoding_layers():
#     kernel = 3
#     filter_size = 64
#     pad = 1
#     pool_size = 2
#     return[
# #         ZeroPadding2D(padding=(pad,pad)),
# #         Conv2D(512, kernel_size=kernel, strides=1, padding='valid'),
# #         BatchNormalization(),

# #         UnPooling2D(poolsize=(pool_size,pool_size)),
# #         UnPooling2D(),
# #         UpSampling2D(),
# #         ZeroPadding2D(padding=(pad,pad)),
# #         Conv2D(256, kernel_size=kernel, strides=1, padding='valid'),
# #         BatchNormalization(),

# #         UnPooling2D(poolsize=(pool_size,pool_size)),
# #         UnPooling2D(),
# #         UpSampling2D(),
#         ZeroPadding2D(padding=(pad,pad)),
#         Conv2D(128,kernel_size=kernel, strides=1, padding='valid'),
#         BatchNormalization(),

# #         UnPooling2D(poolsize=(pool_size,pool_size)),
# #         UnPooling2D(),
#         UpSampling2D(),
#         ZeroPadding2D(padding=(pad,pad)),
#         Conv2D(filter_size, kernel_size=kernel, strides=1, padding='valid'),
#         BatchNormalization(),
#     ]

# autoencoder = Sequential()
# # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
# autoencoder.add(Layer(input_shape=img_shape))

# #autoencoder.add(GaussianNoise(sigma=0.3))
# autoencoder.encoding_layers = create_encoding_layers()
# autoencoder.decoding_layers = create_decoding_layers()
# for l in autoencoder.encoding_layers:
#     autoencoder.add(l)
# for l in autoencoder.decoding_layers:
#     autoencoder.add(l)

# autoencoder.add(Conv2D(2, kernel_size=1, strides=1, padding='valid'))
# autoencoder.add(Reshape((300*400,2)))
# # autoencoder.add(Permute((2, 1)))
# autoencoder.add(Activation('softmax'))
# autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=['accuracy'])
# autoencoder.summary()


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
#     p3 = MaxPooling2D((2, 2))(c3)
#     p3 = Dropout(dropout)(p3)
    
#     c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
#     p4 = MaxPooling2D((2, 2))(c4)
#     p4 = Dropout(dropout)(p4)
    
#     c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
#     u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
#     u6 = concatenate([u6, c4])
#     u6 = Dropout(dropout)(u6)
#     c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
#     u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
#     u7 = concatenate([u7, c3])
#     u7 = Dropout(dropout)(u7)
#     c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c3)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(3, (1, 1), activation='relu')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

input_img = Input((300,400,3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
# In[10]:


model.load_weights('model_unet1.hdf5')


# In[11]:


arr = []
arr.append(X_test)
arr = np.array(arr)
output = model.predict(arr)
print(output.shape)


# In[12]:


# x = np.argmax(output[0],axis=1).reshape((300,400))
# print(x.shape)
# np.place(x,x>0,[255])
# print(x.shape)


# In[13]:


# cv2.imshow('img1',x)
# plt.figure(2)
# cv2.imshow('img2',X_test)
# cv2.imshow('img3',Y_test)
# cv2.waitKey(0)
# plt.imshow(output[0])
# plt.figure(2)
# plt.imshow(X_test)
# plt.show()
# cv2.imwrite('img1.jpg',output[0])
# print(list(output[0]))
# plt.imshow(Y_test)
# cv2.destroyAllWindows()


# In[ ]:




