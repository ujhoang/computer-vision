from pyimagesearch.tranform import four_point_transform
from skimage.filters import threshold_local
import cv2, numpy as np, argparse, imutils

# building the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help= "Path to the image")
args = vars(ap.parse_args())