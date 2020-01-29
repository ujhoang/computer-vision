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
'''
cv2.imwrite("./images/grey tetris.jpg", grey)

# 2. Edge Detection
edge = cv2.Canny(image, 30, 150)
cv2.imshow("edged", edge)
cv2.waitKey(0)
cv2.imwrite("./images/edge tetris.jpg", edge)
'''
# 3. Thresholding
# By thresholding the pixels <225 to 255 (White) - FOREGROUND
# and pixels >=225 to 0 (Black), this segments the image - BACKGROUND
# cv2.threshold[1] an array of pixel greyscale codes that would be used to create the edge detected image
thresh = cv2.threshold(grey, 225, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)
cv2.imwrite("./images/black_white.jpg", thresh)

# 4. Finding, Counting and Drawing Contour
# Separating the background from the foreground is critical to image processing
contours = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("Contours", contours)
cv2.waitKey(0)