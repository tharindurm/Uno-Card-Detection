import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)

print("Starting")
image = cv2.imread('B1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize(image)

print("Middle")
transform = A.ShiftScaleRotate(p=0.5)
random.seed(7) 
augmented_image = transform(image=image)['image']
cv2.imshow("img",augmented_image)
cv2.waitkey(0)
