import cv2
import numpy as np
import os


#Filtering all the jpg files in the current directory
names = []
for file in os.listdir("./"):
    if file.endswith(".jpg"):
        names.append(file)


#Function taken from:
#https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate(image, angle, center = None, scale = 1.1):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


for i in names:        
    img=cv2.imread(i)
    print(img.shape)

    rot = rotate(img,-20,None,1.1)

    cropped_image = rot[1100:3600, 0:]

    #cv2.imshow("sheared",sheared_img)
    cv2.imwrite(str(i),cropped_image)
    print(rot.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#Cropping middle region
for name in names:
    img = cv2.imread(name)
    print("Cropping image:",name)
    cropped_image = img[1100:3600, 0:]
    cv2.imwrite("./Cropped/"+str(name), cropped_image)
'''
