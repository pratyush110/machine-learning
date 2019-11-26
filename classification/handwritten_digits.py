#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:59:31 2019

@author: pratyush
"""

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

"""
dataset of 60000 28x28 grayscale images of the 
10 digits, along with a test set of 10000 images
"""

from keras.datasets import mnist

pixel_width = 28
pixel_height = 28
num_of_classes = 10

"""
num_samples here deals with the color of the image 
since we have grayscale so the num_sample will have 1
"""

(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

"""
returns 2 tuples
feature_train, feature_test->uint8 array of grayscale image data with shape(num_samples,28,28)
labels_train, labels_test->uint8 array of digit labels(ints in 0-9) with shape(num_samples)
"""

#to add depth in the images as CNN model requires depth for the images
features_train = features_train.reshape(features_train.shape[0], pixel_width, pixel_height, 1)
features_test = features_test.reshape(features_test.shape[0], pixel_width, pixel_height, 1)

#CNN model requires these three inputs so it says that all the images have the same dimentions
input_shape = (pixel_width, pixel_height, 1)

#model takes values in the format of float only
features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

#this will convert all the numerical values of pixels into percentage
features_train /= 255
features_test /=255

"""
as we know at the end we get the % value indicating the label it will belong to so the 
current label value which is 2 will not be helpful so we have to flatten 2 into binary
class matrix->[0,0,1,0,0,0,0,0,0,0] so 2 will become the array represnting 2 on third 
position in the array with element 1 and rest with 0""" 
labels_train = keras.utils.to_categorical(labels_train, num_of_classes)
labels_test = keras.utils.to_categorical(labels_test, num_of_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
#print(model.output_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
#print(model.output_shape)
#to minimize the overfitting of the model
#the dropout method will take 25% of data and drop it to make randomness in the dataset
#this makes sure that the model doesnt mug up a certain feature from the image
