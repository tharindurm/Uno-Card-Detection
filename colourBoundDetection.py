import cv2
import numpy as np


#lower = np.array([125,10,20])
#upper = np.array([255,255,255])

img = cv2.imread("1.JPG")


#Scaling
scale_percent = 50 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

combined = np.ones((img.shape[0],img.shape[1]))

#blur = cv2.GaussianBlur(img,(9,9),0)
blur = cv2.bilateralFilter(img,9,75,75)
blur = cv2.medianBlur(blur,5)
mask = blur
HSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


def filter(x):
    mask1_H_L = cv2.getTrackbarPos('Hue-Lower', 'mask1')
    mask1_H_H = cv2.getTrackbarPos('Hue-Higher', 'mask1')

    mask1_S_L = cv2.getTrackbarPos('Sat-Lower', 'mask1')
    mask1_S_H = cv2.getTrackbarPos('Sat-Higher', 'mask1')

    mask1_V_L = cv2.getTrackbarPos('Val-Lower', 'mask1')
    mask1_V_H = cv2.getTrackbarPos('Val-Higher', 'mask1')

    m1_lower = np.array([mask1_H_L,mask1_S_L,mask1_V_L])
    m1_upper = np.array([mask1_H_H,mask1_S_H,mask1_V_H])
    mask1 = cv2.inRange(HSV, m1_lower, m1_upper)

    mask2_H_L = cv2.getTrackbarPos('Hue-Lower', 'mask2')
    mask2_H_H = cv2.getTrackbarPos('Hue-Higher', 'mask2')

    mask2_S_L = cv2.getTrackbarPos('Sat-Lower', 'mask2')
    mask2_S_H = cv2.getTrackbarPos('Sat-Higher', 'mask2')

    mask2_V_L = cv2.getTrackbarPos('Val-Lower', 'mask2')
    mask2_V_H = cv2.getTrackbarPos('Val-Higher', 'mask2')

    m2_lower = np.array([mask2_H_L, mask2_S_L, mask2_V_L])
    m2_upper = np.array([mask2_H_H, mask2_S_H, mask2_V_H])
    mask2 = cv2.inRange(HSV, m2_lower, m2_upper)

    #kernel = np.ones((3,3), np.uint8)
    #mask_dilation = cv2.dilate(mask, kernel, iterations=1)
    #img_erosion = cv2.erode(mask_dilation, kernel, iterations=1)

    #cv2.imshow("mask", mask)
    #0-2,29-168,68-255
    #mask1 = cv2.inRange(HSV, (0,29,68), (2,168,255))

    combined = cv2.add(mask1,mask2)
    #cv2.imshow("mask_combined", combined)
    vis = np.concatenate((mask1,mask2,combined), axis=1)
    cv2.imshow("Concat-x",vis)

    h,s,v = cv2.split(HSV)

    x = h*combined
    cv2.imshow("x",x)
    


#create a seperate window named 'mask(x)' for trackbars
cv2.namedWindow('mask1')
cv2.namedWindow('mask2')

#create trackbar in 'mask1' window
cv2.createTrackbar('Hue-Lower','mask1',132,180,filter)
cv2.createTrackbar('Hue-Higher','mask1',0,180,filter)
cv2.createTrackbar('Sat-Lower','mask1',42,255,filter)
cv2.createTrackbar('Sat-Higher','mask1',168,255,filter)
cv2.createTrackbar('Val-Lower','mask1',0,255,filter)
cv2.createTrackbar('Val-Higher','mask1',245,255,filter)

#create trackbar in 'mask2' window
cv2.createTrackbar('Hue-Lower','mask2',0,180,filter)
cv2.createTrackbar('Hue-Higher','mask2',2,180,filter)
cv2.createTrackbar('Sat-Lower','mask2',66,255,filter)
cv2.createTrackbar('Sat-Higher','mask2',232,255,filter)
cv2.createTrackbar('Val-Lower','mask2',0,255,filter)
cv2.createTrackbar('Val-Higher','mask2',255,255,filter)




cv2.imshow("Original",img)
cv2.waitKey(0)



