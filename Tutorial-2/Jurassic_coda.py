# importing packages
import cv2, imutils

#loading the image, images are stored as (height, width, dimension)
image = cv2.imread("uj.jpg")
(h, w, d) = image.shape
print("width={}, height={}, dimension={}".format(w,h,d))

cv2.imshow("Image", image)
cv2.waitKey(0)
name="ujsaved.jpg"
cv2.imwrite(name, image)
print("{} saved".format(name))