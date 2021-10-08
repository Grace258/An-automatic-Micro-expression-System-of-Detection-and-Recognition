import os
import cv2
import numpy


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers  import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns 
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

os.environ['CUDA_VISIBLE_DEVICES']= '1'

K.image_data_format() == "channels_last"

image_rows, image_columns, image_depth = 64, 64, 88

negativepath = 'train_set/negative/'
positivepath = 'train_set/positive/'
surprisepath = 'train_set/surprise/'
otherpath = 'train_set/others/'

def cv_imread(filepath):
    cv_img = cv2.imdecode(numpy.fromfile(filepath,dtype = numpy.uint8),-1)
    return cv_img

def augment(data):
    temp = data.copy()
    rot = [[],[],[],[]]
    for i in data:
        rotated = i
        for t in range(1,4):
            rotated = list(zip(*rotated[::-1]))
            rot[t-1].append(rotated)
        rot[3].append(numpy.flip(numpy.array(i), 1))
    for i in range(len(rot)):
        rot[i] = numpy.asarray(rot[i])
        rot[i] = numpy.rollaxis(numpy.rollaxis(rot[i], 2, 0), 2, 0)
    return temp, rot

def get_neg_data(data_augmentation):
    neg_list = []
    directorylisting = os.listdir(negativepath)
    count = 0
    for video in directorylisting:
        count = count + 1
        videopath = negativepath + video
        #print(videopath)
        frames = []
        framelisting = os.listdir(videopath)
        framerange = [x for x in range(88)]
        #print(videopath)
        for frame in framerange:
               imagepath = videopath + "/" + framelisting[frame]
               image = cv_imread(imagepath)
               #print(imagepath)
               imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
               #print(image.shape)
               
               grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
               imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
               
               frames.append(grayimage)
        if(data_augmentation):
            frames, roted = augment(frames)
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            neg_list.append(videoarray)
            for r in range(len(roted)):
                neg_list.append(roted[r])
        else:
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            neg_list.append(videoarray)
    return neg_list
        

def get_pos_data(data_augmentation):
    pos_list = []
    directorylisting = os.listdir(positivepath)
    count1 = 0
    for video in directorylisting:
        count1 = count1 + 1
        videopath = positivepath + video
        frames = []
        framelisting = os.listdir(videopath)
        framerange = [x for x in range(88)]
        print(videopath)
        for frame in framerange:
               imagepath = videopath + "/" + framelisting[frame]
               image = cv_imread(imagepath)
               imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
               grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
               imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
               
               frames.append(grayimage)
        if(data_augmentation):
            frames, roted = augment(frames)
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            pos_list.append(videoarray)
            for r in range(len(roted)):
                pos_list.append(roted[r])
        else:
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            pos_list.append(videoarray)
    return pos_list


def get_sur_data(data_augmentation):
    sur_list = []
    directorylisting = os.listdir(surprisepath)
    count2 = 0
    for video in directorylisting:
        videopath = surprisepath + video
        count2 = count2 + 1
        frames = []
        framelisting = os.listdir(videopath)
        framerange = [x for x in range(88)]
        for frame in framerange:
               imagepath = videopath + "/" + framelisting[frame]
               image = cv_imread(imagepath)
               #print(frame)
               #print(imagepath)
               imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
               grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
               imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
               
               frames.append(grayimage)
        if(data_augmentation):
            frames, roted = augment(frames)
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            sur_list.append(videoarray)
            for r in range(len(roted)):
                sur_list.append(roted[r])
        else:
            frames = numpy.asarray(frames)
            videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
            sur_list.append(videoarray)
    return sur_list


'''directorylisting = os.listdir(otherpath)
count3 = 0
for video in directorylisting:
    count3 = count3 + 1
    videopath = otherpath + video
    #print(videopath)
    frames = []
    framelisting = os.listdir(videopath)
    framerange = [x for x in range(18)]
    for frame in framerange:
           imagepath = videopath + "/" + framelisting[frame]
           image = cv_imread(imagepath)
           #print(imagepath)
           imageresize = cv2.resize(image, (image_rows, image_columns), interpolation = cv2.INTER_AREA)
           #print(image.shape)
           grayimage = cv2.cvtColor(imageresize, cv2.COLOR_BGR2GRAY)
           imageresize = imageresize.reshape(1,imageresize.shape[0],imageresize.shape[1],imageresize.shape[2])
           frames.append(grayimage)
    frames = numpy.asarray(frames)
    videoarray = numpy.rollaxis(numpy.rollaxis(frames, 2, 0), 2, 0)
    #videoarray2 = numpy.rollaxis(numpy.rollaxis(frames2, 2, 0), 2, 0)
    training_list.append(videoarray)'''
    #training_list.append(videoarray2)
    #print(training_list)

def load_data():
    temp = []
    neg_data = get_neg_data(data_augmentation = True)
    pos_data = get_pos_data(data_augmentation = True)
    sur_data = get_sur_data(data_augmentation = False)
    training_list = neg_data
    training_list.extend(pos_data)
    training_list.extend(sur_data)
    training_set = numpy.array(training_list).astype('float32')

    trainingsamples = training_set.shape[0]
    traininglabels = numpy.zeros((trainingsamples, ), dtype = int)
    traininglabels[0:345] = 0
    traininglabels[345:695] = 1
    traininglabels[695:] = 2
    #traininglabels[304:] = 3

    traininglabels = to_categorical(traininglabels, 3)
    training_set /= 255.0
    training_set -= 0.5
    # training_set = training_set.reshape(training_set.shape[0],training_set.shape[1], training_set.shape[2], training_set.shape[3], 1)
    
    train_images, test_images, train_labels, test_labels =  train_test_split(training_set, traininglabels, test_size=0.1, random_state=37)
    
    return train_images, test_images, train_labels, test_labels
    # return temp, temp, temp_y, temp_y

# MicroExpSTCNN Model
def build_model_2d():
    model = Sequential()
    initializer = tf.random_normal_initializer(0., 0.02)
    model.add(Conv2D(32, 3, kernel_initializer=initializer, padding="same", activation='linear'
                     , input_shape=(image_rows, image_columns, image_depth)
                     , data_format='channels_last', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Conv2D(64, 3, kernel_initializer=initializer, padding="same", activation='linear'
                    , data_format='channels_last', kernel_regularizer=regularizers.l1(0.01)))
    model.add(MaxPooling2D(3,3))
    model.add(BatchNormalization())
    model.add(Conv2D(128, 3, kernel_initializer=initializer, padding="same", activation='linear'
                     , data_format='channels_last', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Conv2D(256, 3, kernel_initializer=initializer, padding="same", activation='linear'
                     , data_format='channels_last', kernel_regularizer=regularizers.l1(0.01)))
    model.add(MaxPooling2D(3,3))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='linear', kernel_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(1e-04), metrics = ['accuracy'])
    model.summary()
    
    filepath="weights_microexpstcnn/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    tf.keras.utils.plot_model(model, show_shapes=True, dpi=64, to_file='model2.png')
    return model

def save_and_load(model):

    '存模型&讀模型'
    model.save("detect_expression_model.h5")
    #model = load_model('CNN_model')
    
    epochs = range(len(history.history['accuracy']))
    plt.figure()
    plt.plot(epochs, history.history['accuracy'], 'b', label = 'training acc')
    plt.plot(epochs, history.history['val_accuracy'], 'r', label = 'validation acc')
    plt.title('training and validation accurancy')
    plt.legend()
    plt.savefig('second_acc.jpg')
    
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'b', label = 'training loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label = 'validation loss')
    plt.title('training and validation loss')
    plt.legend()
    plt.savefig('second_loss.jpg')
    
    
def confustion(model, test_images, test_labels):
    sns.set() 
    y_true = test_labels
    y_true = numpy.argmax(test_labels, axis=1)
    y_pred = model.predict(test_images)
    y_pred = numpy.argmax(y_pred, axis=1)
    C2= confusion_matrix(y_true, y_pred, labels=[0, 1, 2]) 
    sns.heatmap(C2,annot=True, cmap="YlGnBu")



if __name__ == '__main__':
    
    train_images, test_images, train_labels, test_labels = load_data()

    model = build_model_2d()
    # model = load_model("detect_expression_model.h5")

    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), batch_size = 64, epochs = 100, shuffle=True)
 
    model.evaluate(test_images, test_labels)
    o = model.predict(test_images)
    save_and_load(model)

    confustion(model, test_images, test_labels)
    