import cv2 as cv
import numpy as np
import sys

def get_cont(impath):
    # Read the image and transform it to grayscale
    # impath = 'numbers.jpg'
    img = cv.imread(impath)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply thresholding
    ret, threshold = cv.threshold(img, 141, 255, cv.THRESH_BINARY_INV)

    # Kernel methods: opening, dilation
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(threshold, cv.MORPH_OPEN, kernel)
    dil = cv.dilate(opening, kernel, iterations=3)

    # Contour searching
    cont, hier = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    return cont

def sort_cont(cnt):
    i = 1
    b = [cv.boundingRect(c) for c in cnt]
    (cnt, b) = zip(*sorted(zip(cnt, b),
                            key=lambda b: b[1][i],
                            reverse=False))
    return cnt, b


def write_images(img, cont):

    xdd = []
    ydd = []
    wdd = []
    hdd = []

    for i in cont:
        (xd, yd, wd, hd) = cv.boundingRect(i)
        if (wd >= 0) and (hd >= 100):
            xdd.append(xd)
            ydd.append(yd)
            wdd.append(wd)
            hdd.append(hd)

    numbers = []
    for x, y, w, h in zip(xdd, ydd, wdd, hdd):
        num = img[y:y + h, x:x + w]
        numbers.append(num)
    print(numbers)
    k = 0
    for i in numbers:
        i = cv.resize(i, (28, 28))
        ret, i = cv.threshold(i, 143, 255, cv.THRESH_BINARY_INV)
        cv.imwrite(f'img/{k}.jpg', i)
        k += 1
