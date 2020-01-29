'''
6 objectives
    1. Convert to greyscale
    2. Edge detection
    3. Thresholding a greyscale image
    4. Finding, counting, and drawing contours
    5. Conducting Erosion and dilation
    6. Masking an image
'''

import argparse, cv2, imutils

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help= "Path to put image")
args = vars(ap.parse_args())

# 1. Converting an image to greyscale
image = cv2.imread(args["image"])
cv2.imshow("image",image)
cv2.waitKey(0)

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("grey", grey)
cv2.waitKey(0)

# 2. Edge Detection
edge = cv2.Canny(image, 30, 150)
cv2.imshow("edged", edge)
cv2.waitKey(0)

# 3. Thresholding