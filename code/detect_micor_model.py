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

training_list = []

micropath = 'train_set2/micro/'
nonmicropath = 'train_set2/non_micro/'

def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath,dtype = np.uint8),-1)
    return cv_img

directorylisting = os.listdir(micropath)
count = 0
for video in directorylisting:
    count = count + 1
    videopath = micropath + video
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
    training_list.append(videoarray)   
    

directorylisting = os.listdir(nonmicropath)
count1 = 0
for video in directorylisting:
    count1 = count1 + 1
    videopath = nonmicropath + video
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
    training_list.append(videoarray)

#print(videoarray.shape)
training_list = numpy.asarray(training_list)
trainingsamples = len(training_list)
print(count,count1)
traininglabels = numpy.zeros((trainingsamples, ), dtype = int)

traininglabels[0:267] = 0
traininglabels[267:] = 1

traininglabels = np_utils.to_categorical(traininglabels, 2)

training_data = [training_list, traininglabels]
(trainingframes, traininglabels) = (training_data[0], training_data[1])
training_set = numpy.zeros((trainingsamples, 1, image_rows, image_columns, image_depth))
#print(count,count1,count2)

for h in range(trainingsamples):
    training_set[h][0][:][:][:] = trainingframes[h, :, :, :]

training_set = training_set.astype('float32')
training_set -= numpy.mean(training_set)
training_set /= numpy.max(training_set)

# Spliting the dataset into training and test sets
train_images, test_images, train_labels, test_labels =  train_test_split(training_set, traininglabels, test_size=0.2, random_state=4)
train_images = train_images.reshape(train_images.shape[0], train_images.shape[4], train_images.shape[2], train_images.shape[3], train_images.shape[1])
test_images = test_images.reshape(test_images.shape[0], test_images.shape[4], test_images.shape[2], test_images.shape[3], test_images.shape[1])
# MicroExpSTCNN Model
model = Sequential()
initializer = tf.random_normal_initializer(0., 0.02)
model.add(Conv3D(32, (3, 3, 15), kernel_initializer=initializer, input_shape=(image_depth, image_rows, image_columns,1), data_format='channels_last'))
model.add(tf.keras.layers.Activation('relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(tf.keras.layers.Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))
#model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(0.01), metrics = ['accuracy'])
model.summary()

filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file='model.png')

# Training the model
#history = model.fit(train_images, train_labels, test_data = (test_images, test_labels), callbacks=callbacks_list, batch_size = 16, epochs = 10, shuffle=True)
history = model.fit(train_images, train_labels, validation_data = (test_images, test_labels), batch_size = 16, epochs = 15)
# Finding Confusion Matrix using pretrained weights

#%%
'存模型&讀模型'
model.save("detect_micro_model.h5")
#model = load_model('CNN_model')

epochs = range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs, history.history['accuracy'], 'b', label = 'training acc')
plt.plot(epochs, history.history['val_accuracy'], 'r', label = 'validation acc')
plt.title('training and validation accurancy')
plt.legend()
plt.savefig('first_acc.jpg')

plt.figure()
plt.plot(epochs, history.history['loss'], 'b', label = 'training loss')
plt.plot(epochs, history.history['val_loss'], 'r', label = 'validation loss')
plt.title('training and validation loss')
plt.legend()
plt.savefig('first_loss.jpg')

