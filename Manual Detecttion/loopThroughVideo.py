import cv2
import os
import numpy as np

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, img = cap.read()
    original = img.copy()

    '''

    #Scale down input video stream
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #Convert image to HSV
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

    #Split Hue,Saturation,Value channels in to 3 variables
    (h, s, v) = cv2.split(imghsv)

    #Increasing saturation channel
    #s = s * 1.5
    #s = np.clip(s, 0, 255)
    #imghsv = cv2.merge([h, s, v])
    #img_rgb = cv2.cvtColor(imghsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    #cv2.imshow("Saturated", img_rgb)

    #Converting to grayscale
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Thresholding
    #ret, img_thresh = cv2.threshold(img_rgb, 125, 255, cv2.THRESH_BINARY)
    #cv2.imshow("thersh",img_thresh)

    cv2.putText(img,"Sample Text",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
    '''
    #press q to quit from the loop
    if cv2.waitKey(20) % 0xFF == ord("q"):
        break
    
    cv2.imshow("img",img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
