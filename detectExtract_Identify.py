import cv2
import os
import numpy as np
import copy

# This function reads the images saved as the templates for template matching process
def returnTemplateName(img):
    # A list to store matching score
    score = []
    # Get the list of images in the templates folder
    temps = os.listdir('./templates')
    for t in temps:
        # Read a template image
        template = cv2.imread('templates/' + t, 0)
        # Apply template matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        # Appends score in to the list
        score.append(res)
    # Find the maximum score and the index of the score in 'score' list. that index is used to get the file name.
    # file name is split from '.' to remove file extension. first part isi returned as the name of best match.
    return temps[score.index(max(score))].split(".")[0]


def getMainColor(img):
    # Converting image in to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range of colors to detect
    # Red value range near the START of the spectrum
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([20, 255, 255])

    # Red value range near the END of the spectrum
    _lower_red = np.array([160, 100, 100])
    _upper_red = np.array([180, 255, 255])

    # Green value range
    lower_green = np.array([30, 70, 0])
    upper_green = np.array([90, 255, 110])

    # Blue value range
    lower_blue = np.array([90, 45, 100])
    upper_blue = np.array([140, 255, 255])

    # Yellow value range
    lower_yellow = np.array([15, 45, 123])
    upper_yellow = np.array([35, 255, 255])

    # Create a mask for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red) + cv2.inRange(hsv, _lower_red, _upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Writing msk name on the image
    cv2.putText(mask_red, "Red", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mask_green, "Green", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mask_blue, "Blue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mask_yellow, "Yellow", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Concatenating images vertically to be shown in one frame
    all_masks = cv2.hconcat([mask_red,mask_green,mask_blue,mask_yellow])
    cv2.imshow("Color Masks",all_masks)
    cv2.moveWindow('Color Masks', 500, 90)

    # Calculating number of white pixels in the mask
    red_pixels = np.sum(mask_red == 255)
    green_pixels = np.sum(mask_green == 255)
    blue_pixels = np.sum(mask_blue == 255)
    yellow_pixels = np.sum(mask_yellow == 255)

    # List to store the names of the color categories
    names = ['Red','Green','Blue','Yellow']

    # List to stire the number of white pixels in the corresponding mask
    values = [red_pixels,green_pixels,blue_pixels,yellow_pixels]

    # Return the name of the color where max white pixels were found in corresponding mask
    return names[values.index(max(values))], all_masks


# Start reading video file
cap = cv2.VideoCapture('ca.mp4')

# This variable was used to append at the file name during template collection stage
# num = 0

while cap.isOpened():
    # Reading a frame from the source
    ret, img = cap.read()

    # Reading height and width of the frame
    height, width = img.shape[:2]

    # Resizing image
    # This can be changed according to the preference. Currently, the size is reduced by half
    resized_img = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)
    resized_img_original = resized_img.copy()
    cv2.putText(resized_img_original, "Original", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    resized_img_contour = resized_img.copy()
    cv2.putText(resized_img_contour, "Card contour", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Converting color image in to grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    # Thresholding the grayscale image to get a binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Finding contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("thresh",thresh)

    # If some contours are found
    if len(contours):
        # Finding the largest contour in the image
        # This extracts the card contour as background is already removed by thresholding to leave only the card
        largest_contour = max(contours, key=cv2.contourArea)

        # Drawing detected contour on top the image
        cv2.drawContours(resized_img_contour, largest_contour, -1, (0, 255, 0), 5)

        # Finding coordinates of the box with minimum background other than contour area. Since this is to detect
        # a Uno card which is already a rectangle, it covers all the card with additional background at
        # rounded corners of the card. This returns center point, width, height, and angle of rotation of the box
        # needed to surround the contour
        rect = cv2.minAreaRect(largest_contour)

        # This converts the previously returned information in to 4 corner point coordinates on the image
        box = cv2.boxPoints(rect)

        # Converts the coordinate values in to integers to get rid of floating values
        # introduced during previous functions' calculations
        box = np.int0(box)

        # Draw the rotated rectangle on the image
        cv2.drawContours(resized_img_contour, [box], 0, (0, 0, 255), 2)
        # cv2.imshow("Gray", gray)
        # cv2.imshow("Resized_img", resized_img)

        # Get the rotation angle of the rectangle
        angle = rect[2]

        # This correction was done as some images were rotated horizontally when they were originally tiled to left.
        # In this application the card needs to be upright whether it is tilted towards left or right
        if angle>45:
            angle = angle-90

        # Rotating the image to straighten the rectangle
        # Getting height and width of the image
        (h, w) = resized_img.shape[:2]

        # Calculating the center coordinate of the image
        center = (w // 2, h // 2)

        # Retrieving rotational matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotating the image using previously calculated matrix
        img_rotated = cv2.warpAffine(resized_img, M, (w, h))
        #cv2.imshow("img_rotated",img_rotated)

        ###############
        # Now that the image is rotated to adjust the card in to upright position, previously calculated contour
        # information is no longer valid for the new image. Same information is recalculated for the new image
        ###############

        # Converting rotated image in to grayscale
        img_rotated_gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
        cv2.putText(img_rotated, "Re-Oriented", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Thresholding image to get binary image
        _, img_rotated_thresh = cv2.threshold(img_rotated_gray, 127, 255, cv2.THRESH_BINARY)

        # Finding contours of the new image
        img_rotated_contours, img_rotated_hierarchy = cv2.findContours(img_rotated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Rest of the program will be executed only if contours are found in previous step
        if len(img_rotated_contours):
            # Finding largest contour
            img_rotated_largest_contour = max(img_rotated_contours, key=cv2.contourArea)

            # Now the coordinates for a rectangle to surround the card is calculates instead of cv2.minAreaRect as the
            # image is oriented to make the card straight
            # This returns the x,y coordinates of the center, width and height of the box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Cropping the image to extract only the card
            img_rotated_cropped_image = img_rotated[y:y + h, x:x + w]

            # Resizing the images to view easily.
            tiny_img_rotated_cropped_image = cv2.resize(img_rotated_cropped_image, (128,192), interpolation=cv2.INTER_AREA)
            #cv2.imshow("tor",tiny_img_rotated_cropped_image)

            # Canny edge detection to refine edge features
            edge = cv2.Canny(tiny_img_rotated_cropped_image, 50, 150)
            # Making the image a 3 channel image to make it compatible with other images when displaying
            edge_3_channel = cv2.merge((edge,edge,edge))


            #####
            # Main function to find the name of the card using template matching
            # This function returns the name of best matching template
            name = returnTemplateName(edge)

            #####
            # Calculate the color of the card
            color, masks = getMainColor(tiny_img_rotated_cropped_image)

            # Checks if a color card or a black card is selected
            if name == 'Take4' or name == 'Wild':
                card_name = name
                cv2.putText(resized_img, card_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                card_name = color + " " + name
                cv2.putText(resized_img, card_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            concat_imgs = cv2.hconcat([tiny_img_rotated_cropped_image, edge_3_channel])
            cv2.imshow("Card/Canny", concat_imgs)
            cv2.moveWindow('Card/Canny', 240, 90) # Repositioning window

            big_imgs = cv2.hconcat((resized_img_original,img_rotated,resized_img_contour,resized_img))
            cv2.imshow("Final Output",big_imgs)
            cv2.moveWindow('Final Output', 240, 315)



            '''
            ############
            ### This segment was used to store the templates. 
            ### When the preview is running pressing 'c' save the preview as template
            key = cv2.waitKey(1)
            if key == ord('c'):
                cv2.imwrite(str(num)+".jpg",edge)
                print("Capture saved ================================ ")
                num +=1
            ############
            '''
        # Pauses the loop for 1 millisecond to display the images in windows
        key = cv2.waitKey(1)

        # Break the loop if ESC is pressed
        if key == 27: # Esc key
            break

# Keeps the windows open indefinitely till manually closed
cv2.waitKey(0)
cv2.destroyAllWindows()