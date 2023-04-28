
# Uno Card Recognition using Image Processing (OpenCV-Python)

This project is focussed on recognizing Uno cards. In this implementation a video file is chosen as the input data stream for the program.

<<Whole image>>

## Repository Overview
This repository contains the main program `detectExtract_Identify.py` which can be executed as a normal python program. The `templates` folder contains the images which will be used as the templates in the card recognization process. `video.mp4` is the default input data stream for the program. The `colourLimitIdentification.py` program is used to identify the upper and lower limits of CSV calues which is used in the main program. Once identified the best ranges, these values need to be updated in the program manually. The `'Old Miscellaneous'` folder contains some of the python programs which are NOT organized or error free but used in previous attempts to detect Uno cards using different approaches.

## Execution Instructions 
The python file, video and the templates folder have to be in the same location in order for the program to access video and the templates inside the folder.

 The program can be executed from the terminal using following command.
```shell
  python filename.py
```


## Approach taken to recognize the Uno card

The unique cards in a UNO card set can be identified as follows,

- Numbers from 0 to 9 in 4 different colors (40 unique Cards)
- Take 2 in 4 different colors (4 unique cards)
- Skip in 4 different colors (4 unique cards)
- Reverse in 4 different colors (4 unique cards)
- Wild Take 4 card
- Wild card

A total of 54 unique cards are available for detection and recognition. In order to recognize the card, the image is processed using 2 main function to detect the color of the card and the number or symbol. 

In order to make the matching process accurate and less complex, it was decided to remove the color information from the image and focus only on the number or symbol recognition and recognising the color of the card in a different stage. This reduced the number of unique cards required to recognise down to 15 cards.

### Detecting Numbers and symbols
In order to detect the number or the symbol, template matching was used along with a series of preprocesing techniques. The image was first acquired and converted in to gayscale. This image was later used for thresholding. This removes the dark background from the image while preserving only the uno card.

<<img: ori-gray-des>>

This binary image is then processed to find the largest contour in the image. This outlines the UNO card itself. Instead of having the coordinates for a bounding box to enclose the UNO card, `cv2.minAreaRect()` is used to calculate the  coordinates for a rectangle which fits around the UNO card. This significantly reduces the seeping of background of the image in to the bounding box which could happen if the card is tilted. The `cv2.minAreaRect()` function also return the tilt angle of the rectangle. Using this information, the original color image is then rotated to make the card vertically aligned using `cv2.warpAffine()` function.

By this point the UNO card is properly aligned vertically but previously computed contours and coordinates are no longer valid as the image is rotated. But now that the image is aligned, `cv2.boundingRect()` function is used to get the bounding box around the card. By using the retun values of the function which are, the centerpoint x,y coordinates of the rectangle, width and height information of the boundingbox, the UNO card can be cropped out to completely.

This image is then processed using canny edge detection which outlines the edges of the card. Every unique card was processed according to the mentioned procedure and those images were saved separately to be used as templates when actual template matching is being done. The `returnTemplateName()` function in the main program is responsible for doing the template matching and card or symbol recognition process.


<<Templates images>>


During the card recognition process, saved templates were loaded in to the program and the acquired image is then processed according to above mentioned process. Once it reaches the canny edge detected stage, it is compared with the templates to find the best match using `cv2.matchTemplate()` function.



### Detecting color
`getMainColor()` function is responsible for the detection of color of the card. The following procedure is taken in order to determine the color of the card.

In order to detect color, upper and lower limits for the Red, Green, Blue and Yellow in CSV color space need to be determined. This is done by using `colourLimitIdentification.py` program.

<<col limit image>>

Using these upper,lower limits for each color and HSV image, separate colour masks are created for each color using the `cv2.inRange()` function. Which means if any color specified in the range is available in the HSV image, those pixels will be visibale as white pixels in the created mask. Then the number of white pixels are counted for every mask and the mask with maximum number of white pixels is considered as the color of the image. In this application, if the color of the card is blue, only `mask_blue` mask will contain white pixels while others does not contain any white pixels. These color limits needs to be rechecked if the lighting conditions of the enviroment changes.


## Limitations
- The current implementation can give best results only when in a controlled environment. If the environment is dynamic with varying light conditions and a complex backgrounds, the program will fail.

- The program will correctly detect card only up to a 45 degree of tils of the cards.

- Program cannot detect the cards if they are occluded



## References

 - [Let's play UNO](https://www.letsplayuno.com/news/guide/20181213/30092_732567.html#:~:text=A%20UNO%20deck%20consists%20of,%2C%20yellow%2C%20blue%20and%20green.)
 - [learnopencv.com](https://learnopencv.com/)
 - [https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html](https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html)
 - [https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html](https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html)
 - [https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html](https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html)
  - [https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html](https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html)
## Authors

- [@tharindurm](https://github.com/tharindurm)

