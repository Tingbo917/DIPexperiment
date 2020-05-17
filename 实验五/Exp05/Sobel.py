import math
import cv2
import numpy as np
from scipy import signal


def pascalSmooth(n):
    # 返回n阶的非归一化的高斯平滑算子
    pascalSmooth = np.zeros([1, n], np.float32)
    for i in range(n):
        pascalSmooth[0][i] = math.factorial(n - 1) / (math.factorial(i) * math.factorial(n - 1 - i))
    return pascalSmooth


def pascalDiff(n):  # 在一半之前是逐差法。。后半部分的值和前半部分对应
    # 返回n阶差分算子
    pascalDiff = np.zeros([1, n], np.float32)
    pascalSmooth_previous = pascalSmooth(n - 1)
    for i in range(n):
        if i == 0:
            # 恒等于1
            pascalDiff[0][i] = pascalSmooth_previous[0][i]
        elif i == n - 1:
            pascalDiff[0][i] = pascalSmooth_previous[0][i - 1]
        else:
            pascalDiff[0][i] = pascalSmooth_previous[0][i] - pascalSmooth_previous[0][i - 1]
    return pascalDiff


def getSmoothKernel(n):
    # 返回两个sobel算子
    pascalSmoothKernel = pascalSmooth(n)
    pascalDiffKernel = pascalDiff(n)

    # 水平方向上的卷积核
    sobelKernel_x = signal.convolve2d(pascalSmoothKernel.transpose(), pascalDiffKernel, mode='full')
    # 垂直方向上的卷积核
    sobelKernel_y = signal.convolve2d(pascalSmoothKernel, pascalDiffKernel.transpose(), mode='full')
    return (sobelKernel_x, sobelKernel_y)


def sobel(image, n):
    rows, cols = image.shape
    # 得到平滑算子
    pascalSmoothKernel = pascalSmooth(n)
    # 得到差分算子
    pascalDiffKernel = pascalDiff(n)

    # 与水平方向的sobel核卷积
    # 先进行垂直方向的平滑
    image_sobel_x = signal.convolve2d(image, pascalSmoothKernel.transpose(), mode='same')
    # 再进行水平方向的差分
    image_sobel_x = signal.convolve2d(image_sobel_x, pascalDiffKernel, mode='same')

    # 与垂直方向的sobel核卷积
    # 先进行水平方向的平滑
    image_sobel_y = signal.convolve2d(image, pascalSmoothKernel, mode='same')
    image_sobel_y = signal.convolve2d(image_sobel_y, pascalDiffKernel.transpose(), mode='same')

    return (image_sobel_x, image_sobel_y)


if __name__ == '__main__':
    I = cv2.imread('../images/danghui_yuantu.bmp', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('origin', I)

    # 卷积
    image_sobel_x, image_sobel_y = sobel(I, 7)

    # cv2.imshow('image_sobel_x', image_sobel_x)
    # cv2.imshow('image_sobel_y', image_sobel_y)

    # 平方和的方式展开
    edge = np.sqrt(np.power(image_sobel_x, 2.0) + np.power(image_sobel_y, 2.0))
    # 边缘强度的灰度级显示
    edge = edge / np.max(edge)
    edge = np.power(edge, 1)
    edge = edge * 255
    edge = edge.astype(np.uint8)
    cv2.imshow('edge', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()