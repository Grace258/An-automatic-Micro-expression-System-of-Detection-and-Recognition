import os
import face_recognition
from PIL import Image, ImageDraw
import cv2

image_path = "test_set/in/now/s20_sur_01/image17597115.bmp"
image = face_recognition.load_image_file(image_path)
#print(image)
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="cnn")
print("照片共辨識出 {} 張臉。".format(len(face_locations)))


# 逐一繪製人臉識別的結果
for face_location in face_locations:
    top, right, bottom, left = face_location
    pos = (left, top, right, bottom)
    
print(pos) 
#左上:(left,top)  #右下(right,bottom)
