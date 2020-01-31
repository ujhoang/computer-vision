'''
1. Detect the answer sheet
2. Get a top down view of the answer sheet
3. Extract the set of bubbles
4. Sort the bubbles in a row
5. Determine the marked bubble in a row
6. Compare the marked answer to the answer key
7. Repeat for all questions.
'''

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np 
import argparse
import imutils
import cv2

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help= "Path to the image file")
args = vars(ap.parse_args())

# define the answer key
ANSWER_KEY = {0:1, 1:3, 2:2, 3:3, 4:1}

# load image, greyscale, blur it and edge detect!
image = cv2.imread(args["image"])
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (5,5), 0)   #Gaussian Function smoothes the gradient!
edged = cv2.Canny(blurred, 150, 200)

cv2.imshow("Original", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
