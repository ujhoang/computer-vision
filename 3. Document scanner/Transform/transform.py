import numpy as np 
import cv2


def order_points(pts):
    # 4 coordinates (x,y) with top left index 0 and bottom left index 3
    #MAINTAINING CONSISTENT ORDER OF THE POINTS IS IMPORTANT
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

def four_points_transform(image, pts):
    #obtain a consistent order of the points and unpack them
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # calculating the width of the rect will be the max of difference between
    # (tl and tr) or (bl and br) in x coordinates
    widthA = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    widthB = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    MaxWidth = max(int(widthA), int(widthB))

    # calculating the width of the rect will be the max of difference between
    # (tl and tr) or (bl and br) in x coordinates
    HeightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    HeightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    MaxHeight = max(int(HeightA), int(HeightB))

    # dimension of the image is (MaxWidth, MaxHeight)
    # with the dimension of the top-eye view, construct the destination coordinates
    # Keeping the order of tl, tr, br, bl
    dst = np.array([[0,0],
                    [0, MaxWidth-1],
                    [MaxHeight-1, MaxWidth-1],
                    [MaxHeight-1, 0]], dtype="float32")

    # Compute the perspective transform matrix then apply it to the coordinates
    M = cv2.perspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (MaxWidth, MaxHeight))
    return warped

