from Transform.transform import four_points_transform
from skimage.filters import threshold_local
import cv2, numpy as np, argparse, imutils

# building the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help= "Path to the image")
args = vars(ap.parse_args())

# Document scanner will be 3 step process
# 1. Detect edges - with Canny model
# 2. Use the edges to find the contour -  find_contour(image, mode, method)
# 3. Use perspective transform to obtain top down view

image = cv2.imread(args["image"])
ratio = image.shape[0] /500.0
orig = image.copy()
image = imutils.resize(image, height=500)

grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey = cv2.GaussianBlur(grey, (5,5), 0)
# Use Canny model to find edges
edged = cv2.Canny(grey, 75, 255)

print("STEP 1: Edge Detection")
cv2.imshow("Original Image", image)
cv2.imshow("Edge detection", edged)
cv2.waitKey(0)

# Finding Contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) ==4:
        screenCnt = approx
        break

# Showing the contour
print("Step 2: Find contours of the paper")
cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Step 3: Apply Perspective Transform and Threshold
#apply the four point transform to see the top down view of the imiage
warped = four_points_transform(orig, screenCnt.reshape(4,2)*ratio)


#convert the warped image to greyscale, to threshold it to give the black and white effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255


# Show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)