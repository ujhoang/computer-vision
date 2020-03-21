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

# Detect edges.
# cnts return coordinates of all the edge points
cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# docCnt returns four corner coordinates
docCnt = None
# check at least one contour was found
if len(cnts) > 0:
    # Sort the contours to size of area in descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break

# Apply a top down view of the original image and the greyscale image
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(grey, docCnt.reshape(4,2))

# To grade the documents, we need to do binarisation, a process of thresholding/segmenting
# the foreground from the background image
# Applying Otsu's thresholding method to binarise it

thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Find contours in the thresh and draw on it
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
    # Compute the bounding box of the contour, then use the box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w/float(h)

    if w>= 20 and h >= 20 and ar >=0.9 and ar <=1.1:
        questionCnts.append(c)


cv2.drawContours(paper, np.array(questionCnts), -1, (0, 0,255), 3)
cv2.imshow("Threshold", paper)
cv2.waitKey(0)

questionCnts = contours.sort_contours(questionCnts, method ="top-to-bottom")[0]
correct = 0 

# Each question has 5 possible answers, to loop over the question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # Sort the contours for the current question from
    # left to right, then initialise the index of the 
    # bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i +5])[0]
    bubbled = None

# Next step is to determine which bubble is filled in
for (j, c) in enumerate(cnts):
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask = cv2.bitwise_and(thresh, thresh, mask= mask)
    total = cv2.countNonZero(mask)

    if bubbled is None or total > bubbled[0]:
        bubbled = (total, j)

color = (0, 0, 255)
k = ANSWER_KEY[q]

if k== bubbled[1]:
    color = (0, 255, 0)
    correct += 1

cv2.drawContours(paper, [cnts[k]], -1, color, 3)
score = correct/5.0 * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
cv2.imwrite("Result.jpg", paper)