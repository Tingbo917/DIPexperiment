# -*- coding: utf-8 -*-
# @Time    : 2020/4/28 10:08 下午
# @Author  : Liu minxuan
# @Email   : liuminxuan2016@gmail.com
# @File    : TheGaussianSmooth.py
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def gaussBlur(image, sigma, H, W, _boundary='fill', _fillvalue=0):
    gaussKenrnel_x = cv.getGaussianKernel(sigma, W, cv.CV_64F)
    gaussKenrnel_x = np.transpose(gaussKenrnel_x)
    gaussBlur_x = signal.convolve2d(image, gaussKenrnel_x, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    gaussKenrnel_y = cv.getGaussianKernel(sigma, H, cv.CV_64F)
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKenrnel_y, mode='same', boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy


image = cv.imread("./images/smooth/danghui_gaosi.bmp", cv.IMREAD_GRAYSCALE)
cv.imshow("image", image)
# H,W=image.shape
blurImage = gaussBlur(image, 5, 51, 51, 'symm')
blurImage = np.round(blurImage)
blurImage = blurImage.astype(np.uint8)
cv.imshow("blurImage", blurImage)
cv.waitKey(0)
cv.destroyAllWindows()
