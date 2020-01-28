# importing packages
import cv2, imutils
import time 
import matplotlib.pyplot as plt

name="uj.jpg"

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
'''
roi = image[80:270, 130:280]
cv2.imshow("Region of Interest Dr", roi)
cv2.waitKey(0)
cv2.imwrite("cropped.jpg", roi)
'''
#Resizing images
#In deep learning we resize images, ignoring aspect ratios, 
#so that the volume fits into a network which requires
#the image to be square and to be of certain dimesion
'''
resize = imutils.resize(image, width=300)
cv2.imshow("resized uj", resize)
cv2.waitKey(0)
cv2.imwrite("resized.jpg", resize)


ratio = 200/w
dim = (200, int(h*ratio))
resize = cv2.resize(image, dim)
cv2.imshow("resized photo", resize)
cv2.waitKey(0)
'''

# Rotating Images
'''
rotated = imutils.rotate(image, 45)
cv2.imshow("rotated uj", rotated)
cv2.waitKey(0)
cv2.imwrite("rotated.jpg", rotated)
'''

#Rotating image within boundary using imutils
'''
rotate_bound = imutils.rotate_bound(image, -45)
cv2.imshow("rotated uj", rotate_bound)
cv2.waitKey(0)
cv2.imwrite("rotated in boundary.jpg", rotate_bound)
'''

# Skeletonize the picture (edge detection)
'''
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
skeleton = imutils.skeletonize(grey, size=(3,3))
cv2.imshow("matplot",skeleton)
cv2.waitKey(0)
cv2.imwrite("skeleton.jpg", skeleton)
'''
