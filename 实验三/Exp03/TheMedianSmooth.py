# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 10:16 下午
# @Author  : Liu minxuan
# @Email   : liuminxuan2016@gmail.com
# @File    : TheMedianSmooth.py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def medianBlur(image, winSize):
    rows, cols = image.shape
    winH, winW = winSize
    halfWinH = (winH - 1) // 2
    halfWinW = (winW - 1) // 2
    medianBlurImage = np.zeros(image.shape, image.dtype)
    for r in range(rows):
        for c in range(cols):
            if r - halfWinH < 0:
                rTop = 0
            else:
                rTop = r - halfWinH
            if r + halfWinH > rows - 1:
                rBottom = rows - 1
            else:
                rBottom = r + halfWinH
            if c - halfWinW < 0:
                cLeft = 0
            else:
                cLeft = c - halfWinW
            if c + halfWinW > cols - 1:
                cRight = cols - 1
            else:
                cRight = c + halfWinW
            region = image[rTop:rBottom + 1, cLeft:cRight + 1]
            medianBlurImage[r][c] = np.median(region)
    return medianBlurImage


image = cv.imread("./images/smooth/xust_jiaoyan.bmp", cv.IMREAD_GRAYSCALE)
cv.imshow("image", image)
blurImage = medianBlur(image, (3, 3))
blurImage = np.round(blurImage)
blurImage = blurImage.astype(np.uint8)
cv.imshow("blurImage", blurImage)
cv.waitKey(0)
cv.destroyAllWindows()
