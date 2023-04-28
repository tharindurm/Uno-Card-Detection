import cv2
import numpy as np
#import pandas
#import os
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from matplotlib.pyplot import imread,imshow,subplot,show

img = cv2.imread("blue-0.jpg")
rows,cols,ch = img.shape

pts1 = np.float32([[0,0],[91,0],[0,137],[91,137]])
pts2 = np.float32([[40,30],[91,50],[0,107],[50,140]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(91,137))

cv2.imshow("win",dst)
cv2.imwrite("s.jpg",dst)




'''


img = cv2.imread("blue-0.jpg")

data_generator = ImageDataGenerator(rotation_range=90, brightness_range=(0.5,1.5),shear_range=(0,45))
data_generator.fit(img)

image_iterator = data_generator.flow(img)

for i in image_iterator:
    cv2.imshow("image",i)
    cv2.waitKey(0)
'''
