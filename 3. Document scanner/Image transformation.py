from Transform.transform import four_points_transform
import numpy as np 
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to image")
ap.add_argument("-c", "--coord", required=True, help="Coordinates of the ROI")

args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(eval(args["coord"]), dtype = "float32")

warped = four_points_transform(image, pts)
cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
