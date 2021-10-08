import os
import cv2
import numpy
import imageio
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
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
os.environ['CUDA_VISIBLE_DEVICES']= '1'

K.image_data_format() == "channels_last"

image_rows, image_columns, image_depth = 64, 64, 88

training_list = []

negativepath = 'train_set/negative/'
positivepath = 'train_set/positive/'
surprisepath = 'train_set/surprise/'

directorylisting = os.listdir(surprisepath)

image_gen = ImageDataGenerator(             
    #rotation_range = 30   
    #width_shift_range=.15,        # 隨機水平移動 ±15%
    #zoom_range=0.5
    horizontal_flip=True 
)

directorylisting = os.listdir(surprisepath)

for video in directorylisting:
    videopath = surprisepath + video
    framelisting = os.listdir(videopath)
    l = []
    framerange = [x for x in range(88)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           img = load_img(imagepath)
           x = numpy.asarray(img, 'f')
           #x = img
           l.append(x)
           #print(frame)
           #print(l.shape)
    l = numpy.array(l)
    #print(l.shape)
    #l = numpy.expand_dims(l, axis = 0)
    savingpath = "sur_rot_30/ho30_" + video
    os.mkdir(savingpath)
    i = 0
    for batch in image_gen.flow(l, batch_size = 88, save_to_dir = savingpath, save_format = 'jpg'):
        i += 1
        if i > 0:
            break;
    


