import cv2
import numpy as np
import imutils
import argparse

# method to read image passed in command line and display it.
def readImage(image):
    # read the image
    image = cv2.imread(image)
    # display image and close after user clicks any key.
    cv2.imshow("Original doc", image)
    # press any key to close all cv2 windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image

# method to process image and display image in each step.
def prepareImage(image):
    # convert image to gray scale to remove colour noise.
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurring the image prevents unnecessary edges from being picked up.(3,3) is blurring kernel size.
    grayImageBlur = cv2.blur(grayImage,(3,3))
    # canny algorithm to detect edges. The two numbers are minVal and maxVal
    # between which it will be chosen as edge if connected to part above maxVal.
    edgeImage = cv2.Canny(grayImageBlur, 100, 250)
    # show the gray, blurred and edge image
    cv2.imshow("GrayScale doc", grayImage)
    cv2.imshow("Blurred doc", grayImageBlur)
    cv2.imshow("Edged doc", edgeImage)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    return edgeImage

# method to draw contour of the main part.
def contourImage(image, edgeImage):
    # find all contours in the edged image and modifies the original image. Hence passing a copy.
    # RETR_LIST retrieves all of the contours without any hierarchical relationships.
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points.
    contours = cv2.findContours(edgeImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    # sort contours area wise and keep highest area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    # find the perimeter of passed countour which is closed(True)
    perimeter = cv2.arcLength(contours[0], True) 
    # find a polygon of specified contour, approximation accuracy, closed.
    dimensions = cv2.approxPolyDP(contours[0], 0.03*perimeter, True)
    # show the contour on image, the first countour, colour and thickness of border.
    cv2.drawContours(image, [dimensions], 0, (10,100,255), 3)
    cv2.imshow("Contoured doc", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dimensions

def widthHeight(rect):
    (tl, tr, br, bl) = rect
    # compute width of document
    widthTop = np.sqrt( (tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    widthBottom = np.sqrt( (bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    width = max(int(widthTop), int(widthBottom))
    # compute height of document
    heightTop = np.sqrt( (tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightBottom = np.sqrt( (tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    height = max(int(heightTop), int(heightBottom))
    return (width, height)

def cornerCoordinates(dimensions):
    dimensions = dimensions.reshape(4,2)
    corners = np.zeros((4,2), dtype="float32")
    # top left corner will have the smallest sum, 
    # bottom right corner will have the largest sum
    # top-right will have smallest difference
    # botton left will have largest difference
    s = np.sum(dimensions, axis=1)
    d = np.diff(dimensions, axis=1)
    corners[0] = dimensions[np.argmin(s)]
    corners[2] = dimensions[np.argmax(s)]
    corners[1] = dimensions[np.argmin(d)]
    corners[3] = dimensions[np.argmax(d)]
    width, height = widthHeight(corners)
    return (width, height, corners)

def scannedImage(image, width, height, corners):
    # dimension of the new image
    newCorners = np.array([ [0,0], [width-1, 0], [width-1, height-1], [0, height-1] ], dtype="float32")
    # compute the perspective transform matrix 
    transformer = cv2.getPerspectiveTransform(corners, newCorners)
    # transform dimensions of document
    finalImage = cv2.warpPerspective(image, transformer, (width, height))
    # final scanned document
    cv2.imshow("Scanned doc",finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# construct the argument parser and parse the arguments
aP = argparse.ArgumentParser()
aP.add_argument("-i", required = True, help = "Path of image to be scanned")
arguments = vars(aP.parse_args())

# method calls to do different parts of the scanning.
image = readImage(arguments["i"])
edgeImage = prepareImage(image)
dimensions = contourImage(image, edgeImage)
width, height, corners = cornerCoordinates(dimensions)
scannedImage(image, width, height, corners)