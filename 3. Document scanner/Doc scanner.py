from Transform.transform import four_points_transform
from skimage.filters import threshold_local
import cv2, numpy as np, argparse, imutils

# building the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help= "Path to the image")
args = vars(ap.parse_args())

# Document scanner will be 3 step process
# 1. Detect edges
# 2. Use the edges to find the contour 
# 3. Use perspective transform to obtain top down view

image = cv2.imread(args["image"])
orig = image.copy()
image = imutils.resize(image, height=300)


grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey = cv2.GaussianBlur(grey, (5,5), 0)
# Use Canny model to find edges
edged = cv2.Canny(grey, 150, 255)

cv2.imshow("Original Image", image)
cv2.imshow("Edge detection", edged)
cv2.waitKey(0)

