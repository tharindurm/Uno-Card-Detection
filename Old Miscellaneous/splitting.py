#This is comment
import cv2
img = cv2.imread("uno.jpg")

height, width, channels = img.shape

print(height,"-",width)

card_width = int(width/14) #14 is the #of cards in a row
card_height = int(height/8) #8 is the number of rows in the image

for y in range(0,height,card_height):
    card_row = img[y:y+card_height]
    for x in range(0,width,card_width):
        card = card_row[:,x:x+card_width]
        print(x," : ",x+card_width)
        cv2.imshow("card",card)
        cv2.imwrite(str(y+x)+".jpg",card)
        cv2.waitKey(0)
