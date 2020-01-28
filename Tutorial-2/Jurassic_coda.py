# importing packages
import cv2, imutils
import time 

name="ujsaved.jpg"

#loading the image, images are stored as (height, width, dimension)
image = cv2.imread(name)
(h, w, d) = image.shape
print("width={}, height={}, dimension={}".format(w,h,d))
cv2.imshow("Image", image)
cv2.waitKey(0)

# saving the image after processing it
cv2.imwrite(name, image)
print("{} saved".format(name))

# access the (B, G, R) pixel in x=50 and y=100
(B, G, R) = image[100, 50]
print("B={}, G={}, R={}".format(B, G, R))

