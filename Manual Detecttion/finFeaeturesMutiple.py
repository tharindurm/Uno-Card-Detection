import cv2
import os
import numpy as np

path = 'Images'
images = []
classNames = []
orb = cv2.ORB_create(nfeatures=400)

#####   Loading Images
myList = os.listdir(path)
print("Total Classes: "+str(len(myList)))

for cl in myList:
    imgCurrent = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCurrent)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

##### Function to extract features from an image
def findSourceDes(images):
    descriptorList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        descriptorList.append(des)
    return descriptorList


def findCardID(img, desList, thresh = 20):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
        print(matchList)
    except:
        pass

    if len(matchList)!=0:
        if max(matchList) > thresh:
            finalVal = matchList.index(max(matchList))
    return finalVal



##### Extrcating features from the images in Images folder
desList = findSourceDes(images)
print(len(desList))

cap = cv2.VideoCapture("diffCol.mp4")
#for i in range(500):
#    cap.read()
#    pass

while True:
    ret, img = cap.read()
    original = img.copy()

    #Scale down input video stream
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #Convert image to HSV
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(imghsv)
    s = s * 1.5
    s = np.clip(s, 0, 255)
    imghsv = cv2.merge([h, s, v])
    img_rgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    cv2.imshow("Saturated", img_rgb)

    #Converting to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Thresholding
    ret, img_thresh = cv2.threshold(img_rgb, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("thersh",img_thresh)





    #Showing correct class
    id = findCardID(img_gray,desList)
    if id != -1:
        cv2.putText(img,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

    cv2.imshow("img",img)
    cv2.waitKey(1)




'''
import cv2

img1 = cv2.imread('Images/1B.jpg')
img2 = cv2.imread('Images/1BCL.jpg')



kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)






print(len(good))

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#imgKp = cv2.drawKeypoints(img1,kp1,None)

cv2.imshow('img2',img2)
cv2.imshow('img1',img1)
cv2.imshow('img3',img3)
cv2.waitKey(0)
'''