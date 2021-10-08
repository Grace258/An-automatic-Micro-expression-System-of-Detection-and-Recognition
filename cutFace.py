import os
import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np

def cv_imread(filepath):
    cv_img = cv2.imdecode(np.fromfile(filepath,dtype = np.uint8),-1)
    return cv_img

dirpath = "./test_set/in/now/"
directory = os.listdir(dirpath)
#print(directory)
for path in directory:
    frames = os.listdir(dirpath+path)
    count = 0
    os.mkdir('in/' + path)
    for frame in frames:
        image = cv_imread(dirpath + path + '/' + frame)
        #print(dirpath+path+'/'+frame)
        x = 104
        y = 104
        w = 308-104
        h = 308-104# + 50
        crop_img = image[y:y+h, x:x+w]
        savingpath = 'in/' + path + '/' + frame 
        cv2.imwrite(savingpath, crop_img)



