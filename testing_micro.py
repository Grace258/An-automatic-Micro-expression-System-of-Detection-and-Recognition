import os
import cv2
import numpy
import imageio
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from keras import backend as K
import sys
import tensorflow as tf
from keras.models import load_model
import glob
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import matplotlib.pyplot as plt
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES']= '1'

K.image_data_format() == "channels_last"

image_rows, image_columns, image_depth = 64, 64, 88

testing_list = []

testpath = 'test_set/in/'

def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath,dtype = np.uint8),-1)
    return cv_img

directorylisting = os.listdir(testpath)
count = 0
for video in directorylisting:
    count = count + 1
    videopath = testpath + video
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(88)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           #print(imagepath)
           image = cv_imread(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    #print(frames.shape)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    testing_list.append(videoarray)   
    

#print(videoarray.shape)
testing_list = numpy.asarray(testing_list)
trainingsamples = len(testing_list)
#print(count,count1)
testing_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))

#print(count,count1,count2)
for h in range(trainingsamples):
    testing_set[h][0][:][:][:] = testing_list[h, :, :, :]

testing_set = numpy.array(testing_set).astype('float32')
testing_set /= 255.0
testing_set -= 0.5

testing_set = testing_set.reshape(testing_set.shape[0], testing_set.shape[4], testing_set.shape[2], testing_set.shape[3], testing_set.shape[1])
# Spliting the dataset into training and test sets

#%%
'存模型&讀模型'
#model.save("detect_micro_model.h5")
model = load_model('detect_micro_model.h5')
o = model.predict(testing_set)

print(o)
print(np.argmax(o,axis = 1))
