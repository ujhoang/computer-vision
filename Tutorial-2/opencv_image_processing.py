# importing packages
import cv2, imutils
import time 
import matplotlib.pyplot as plt

name="./images/uj.jpg"

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
cv2.imwrite("./images/cropped.jpg", roi)
'''
#Resizing images
#In deep learning we resize images, ignoring aspect ratios, 
#so that the volume fits into a network which requires
#the image to be square and to be of certain dimesion
'''
resize = imutils.resize(image, width=300)
cv2.imshow("resized uj", resize)
cv2.waitKey(0)
cv2.imwrite("./images/resized.jpg", resize)


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
cv2.imwrite("./images/rotated.jpg", rotated)
'''

#Rotating image within boundary using imutils
'''
rotate_bound = imutils.rotate_bound(image, -45)
cv2.imshow("rotated uj", rotate_bound)
cv2.waitKey(0)
cv2.imwrite("./images/rotated in boundary.jpg", rotate_bound)
'''

# Skeletonize the picture (edge detection)
'''
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
skeleton = imutils.skeletonize(grey, size=(3,3))
cv2.imshow("matplot",skeleton)
cv2.waitKey(0)
cv2.imwrite("./images/skeleton.jpg", skeleton)
'''

# Smoothing images
# In many image processing, we need to reduce the high frequency noise,
# making it easier for our algorithm to understand the contents rather 
# than the noise which will confuse the algorithm. Blurring an image is
# one of the easiest way of doing so!
'''
blurred = cv2.GaussianBlur(image, (11,11), 10,0,10)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)
cv2.imwrite("./images/blurred uj.jpg", blurred)
'''

# Drawing on an image
# Note when drawing on an image, we are drawing in place
# Hence, it is a good practice to create a copy (image.copy())
# when drawing so we do not destroy the original image

'''
output = image.copy()
cv2.rectangle(output, (140,90), (280,280),(0,0,255), 5)
cv2.imshow("output rectangle", output)
cv2.waitKey(0)
cv2.imwrite("./images/uj_rectangle.jpg", output)
'''
'''
output = image.copy()
cv2.circle(output, (140,280), 40, (50,255,50), -1) # -1 makes the circle solid fill
cv2.imshow("circle", output)
cv2.waitKey(0)
cv2.imwrite("./images/uj_circle.jpg", output)
'''

output = image.copy()
cv2.putText(output, "Yoo Je Hoang, Data Scientist", (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255),1)
cv2.imshow("uj with text", output)
cv2.waitKey(0)
cv2.imwrite("./images/uj_text.jpg", output)
