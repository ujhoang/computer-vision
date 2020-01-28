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

# access the (B, G, R) pixel in x=50 and y=100
(B, G, R) = image[100, 50]
print("B={}, G={}, R={}".format(B, G, R))

#Array slicing and cropping
#extract region of interest starting from (320,60) to (420,160)
roi = image[80:270, 130:280]
cv2.imshow("Region of Interest Dr", roi)
cv2.waitKey(0)
cv2.imwrite("cropped.jpg", roi)

#Resizing images
#In deep learning we resize images, ignoring aspect ratios, 
#so that the volume fits into a network which requires
#the image to be square and to be of certain dimesion

resize = cv2.resize(image, (200,200))
cv2.imshow("resized photo", resize)
cv2.waitKey(0)
cv2.imwrite("resized.jpg", resize)
