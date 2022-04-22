import cv2 as cv
import numpy as np


def sort_cont(cont, method='left-to-right'):
    reverse = False
    i = 0

    if method == 'right-to-left' or 'left-to-right':
        reverse = True

    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1

    boxes = [cv.boundingRect(c) for c in cont]
    (cont, boxes) = zip(*sorted(zip(cont, boxes),
                                key=lambda b: b[1][i],
                                reverse=reverse))
    return cont, boxes


class ImageProcessing():
    def __init__(self, impath):
        self.impath = impath

    def improcess(self, impath):

        # Read the image and transform it to grayscale
        img = cv.imread(impath)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply thresholding
        ret, threshold = cv.threshold(img, 90, 255, cv.THRESH_BINARY_INV)

        # Kernel methods: opening, dilation
        kernel = np.ones((3, 3), np.uint8)
        opening = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel)
        dil = cv.dilate(opening, kernel, iterations=15)

        # Contour searching
        cont, hier = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        return cont





