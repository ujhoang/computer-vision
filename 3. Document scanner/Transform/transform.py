import numpy as np 
import cv2
pts = np.array([(1,1), (0,1), (1,0), (0,0)])

def order_points(pts):
    # 4 coordinates (x,y) with top left index 0 and bottom left index 3
    rect = np.zeros((4,2), dtype= "float32")

    #top left coordinates will have the smallest sum
    #bottom right will have the largest
    s = np.sum(pts ,axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    #top right will have the largest (y - x) diff
    #bottom left will have the smallest (y - x) diff
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmax(diff)]
    rect[3 ]= pts[np.argmin(diff)]

    return rect


print(order_points(pts))


    