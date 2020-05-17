# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 10:11 下午
# @Author  : Liu minxuan
# @Email   : liuminxuan2016@gmail.com
# @File    : TheMeanSmooth.py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def integral(image):
    rows, cols = image.shape
    inteImageC = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            if c == 0:
                inteImageC[r][c] = image[r][c]
            else:
                inteImageC[r][c] = inteImageC[r][c - 1] + image[r][c]
    inteImage = np.zeros(image.shape, np.float32)
    for c in range(cols):
        for r in range(rows):
            if r == 0:
                inteImage[r][c] = inteImageC[r][c]
            else:
                inteImage[r][c] = inteImage[r - 1][c] + inteImageC[r][c]
    inteImage_0 = np.zeros((rows + 1, cols + 1), np.float32)
    inteImage_0[1:rows + 1, 1:cols + 1] = inteImage
    return inteImage_0


def fastMeanBlur(image, winSize, borderType=cv.BORDER_DEFAULT):
    halfH = (winSize[0] - 1) // 2
    halfW = (winSize[1] - 1) // 2
    ratio = 1.0 / (winSize[0] * winSize[1])
    paddImage = cv.copyMakeBorder(image, halfH, halfH, halfW, halfW, borderType)
    paddIntegral = integral(paddImage)
    rows, cols = image.shape
    meanBlurImage = np.zeros(image.shape, np.float32)
    r, c = 0, 0
    for h in range(halfH, halfH + rows, 1):
        for w in range(halfW, halfW + cols, 1):
            meanBlurImage[r][c] = ratio * (
                        paddIntegral[h + halfH + 1][w + halfW + 1] + paddIntegral[h - halfH][w - halfW] -
                        paddIntegral[h + halfH + 1][w - halfW] - paddIntegral[h - halfH][w + halfW + 1])
            c += 1
        r += 1
        c = 0
    return meanBlurImage


image = cv.imread("./images/smooth/danghui_chengxing.bmp", cv.IMREAD_GRAYSCALE)
cv.imshow("image", image)
blurImage = fastMeanBlur(image, (11, 11))
blurImage = np.round(blurImage)
blurImage = blurImage.astype(np.uint8)
cv.imshow("blurImage", blurImage)
cv.waitKey(0)
cv.destroyAllWindows()
